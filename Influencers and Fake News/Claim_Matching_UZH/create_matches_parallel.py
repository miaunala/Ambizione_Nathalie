from datetime import timedelta
import pandas as pd
from joblib import Parallel, delayed, Memory
from tqdm import tqdm
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
import os

# Load NLTK for tokenisation later
nltk.download("punkt")
nltk.download("punkt_tab")


# Load files from before
# for local
base_path = "/Users/nathalieguibert/Desktop/ResAss_Klüser_FS25/Ambizione Nathalie/Influencers and Fake News/Claim_Matching_UZH/data/"
claims_csv = pd.read_csv(base_path + "analysis_claims_df.csv")
posts_csv = pd.read_csv(base_path + "analysis_posts_df.csv")

'''
# for cluster
claims_csv = pd.read_csv("data/data/analysis_claims_df.csv")
posts_csv = pd.read_csv("data/data/analysis_posts_df.csv")
'''

print(f"Initial claims count: {len(claims_csv)}")
print(f"Initial posts count: {len(posts_csv)}")

### Duplicates Handling

## CLAIMS
# Drop full-row duplicates
claims_csv = claims_csv.drop_duplicates()
print(f"After dropping full-row duplicates: {len(claims_csv)}")

# Drop duplicate statements by different authors; order by statement date and only keep the first
claims_csv = claims_csv.sort_values(by="statement_date", ascending=False)
claims_csv = claims_csv.drop_duplicates(subset="statement", keep="first")
print(f"After dropping duplicates by different authors {len(claims_csv)}")

print(f"After deduplication on claims count: {len(claims_csv)}")

## POSTS
# Drop full_row duplicates
posts_csv = posts_csv.drop_duplicates()
print(f"After dropping full-row duplicates: {len(posts_csv)}")



# Converting dates in both datasets to usable datetime
claims_csv["statement_date"] = pd.to_datetime(claims_csv["statement_date"])
posts_csv["post_created"] = pd.to_datetime(posts_csv["post_created"])


##### TOKEN SIMILARITY: BM25 ######
# Load pretrained Sentence Transformer Model
# Standard Model which has also been used in the Claim Matching paper

# Set top_k matches and filter claims to 3-month window around post dates
# to avoid having to process claims without any available posts
top_k = 1000
days_window = 28
min_post_date = posts_csv["post_created"].min() - pd.DateOffset(months=3)
max_post_date = posts_csv["post_created"].max() + pd.DateOffset(months=3)
claims_csv = claims_csv[
    (claims_csv["statement_date"] >= min_post_date) &
    (claims_csv["statement_date"] <= max_post_date)
].reset_index(drop=True)


# Assign post_id to join information of posts later (to not overcrowd caching)
posts_csv = posts_csv.reset_index(drop=True)
posts_csv["post_id"] = posts_csv.index

# Setup cache
memory = Memory("data/joblib_cache", verbose=1)

# Cached Function in Memory
# Process one entire claim
@memory.cache
def process_one_claim_full(verdict, claim_text, claim_date, author_claim, matching_posts_df, top_k):
    print(f"\nProcessing claim: {claim_text[:80]!r}")
    print(f"Claim date: {claim_date.strftime('%Y-%m-%d')}, Matching posts in timeframe: {len(matching_posts_df)}")

    # Tokenise claim  (with NLTK)
    tokenised_claim = [word.lower() for word in word_tokenize(claim_text)]

    # Collect tokenised posts
    tokenised_posts = []
    # Store original rows to retrieve metadata (e.g. post_id) after BM25 scoring
    original_rows = []

    # Tokenise matching posts
    for _, row in matching_posts_df.iterrows():
        text = str(row["text_analysis"])
        tokens = [word.lower() for word in word_tokenize(text)]
        if tokens:
            tokenised_posts.append(tokens)
            original_rows.append(row)
    # (if there are not tokenised posts, return empty list -> should not happen)
    if not tokenised_posts:
        return []

    # Compute BM25 relevance scores between the tokenised claim and each tokenised post
    bm25 = BM25Okapi(tokenised_posts)
    scores = bm25.get_scores(tokenised_claim)

    # If all the scores showcased no relation -> return empty list (should not happen)
    if max(scores) == 0.000:
        return []

    # Get indices with top k scores
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    # Initialise a result list and a set to track seen post IDs
    results = []
    seen_post_ids = set()

    # Iterate through top indices and append every unseen post to the results with meta information
    for i in top_indices:
        post_id = int(original_rows[i]["post_id"])
        if post_id in seen_post_ids:
            continue
        seen_post_ids.add(post_id)

        # Keeping all the relevant post information
        results.append({
            "verdict": verdict,
            "claim_text": claim_text,
            "author_claim": author_claim,
            "post_text": original_rows[i]["text_analysis"],
            "post_id": post_id,
            "score": float(scores[i])
        })
    return results


# Define output paths for both directions
output_path_r1 = "data/data/flat_pairs_bm25_r1.csv"  # [claim_date-28d, claim_date]
output_path_r2 = "data/data/flat_pairs_bm25_r2.csv"  # [claim_date, claim_date+28d]

# Remove old outputs of both directions (if needed)
'''
for _p in (output_path_r1, output_path_r2):
    if os.path.exists(_p):
        os.remove(_p)
        print(f"Old output file deleted: {_p}")
'''

########## Direction r1 ##########
# Direction: Claims - Posts & Time: 28 Days before claim up to date of claim

print("\n[r1] Matching posts in window: [claim_date - 28 days, claim_date]")

# Parallel processing of function process one claim r1
# Run claim–post matching in parallel across all CPU cores
# delayed wraps the function for lazy execution, Parallel(n_jobs=-1) runs it concurrently
results_r1 = Parallel(n_jobs=-1)(
    delayed(process_one_claim_full)(
        row.verdict,
        row.statement,
        row.statement_date,
        row.statement_originator,
        posts_csv[
            (posts_csv["post_created"] >= row.statement_date - pd.Timedelta(days=days_window)) &
            (posts_csv["post_created"] <= row.statement_date)
        ],
        top_k
    )
    # Get tqdm bar to visualise the process
    for row in tqdm(claims_csv.itertuples(index=False), total=len(claims_csv), desc="Matching claims (r1)")
)

# Flatten and merge results
flat_records_r1 = [record for sublist in results_r1 for record in sublist]
flat_df_r1 = pd.DataFrame(flat_records_r1)

# Join with all post variables
full_output_r1 = flat_df_r1.merge(posts_csv, on="post_id", how="left")
print(f"[r1] Pairs after merge: {len(full_output_r1)}")

# Dupe logic (just if needed)
# Deduplicate on claim + post, Logic Time Phase 1
'''
if not full_output_r1.empty:
    dupes_r1 = full_output_r1[full_output_r1.duplicated(subset=["claim_text", "post_id", "author_claim"], keep=False)]
    print("[r1] Duplicate claim–post pairs found:", len(dupes_r1))
    if not dupes_r1.empty:
        dupes_r1.to_csv("dupe_debug_r1.csv", index=False)
        print("[r1] Duplicates written to dupe_debug_r1.csv")
else:
    print("[r1] No pairs found; nothing saved.")
'''

# Save output to predefined output path
full_output_r1.to_csv(output_path_r1, index=False)
print(f"[r1] Saved to {output_path_r1} with {len(full_output_r1)} claim–post pairs.")



########## Direction r2 ##########
# Direction: Claim - Posts & Time: Date of Claim and 28 days after
print("\n[r2] Matching posts in window: [claim_date, claim_date + 28 days]")

# Parallel processing of function process one claim r2
# Run claim–post matching in parallel across all CPU cores
# delayed wraps the function for lazy execution, Parallel(n_jobs=-1) runs it concurrently
results_r2 = Parallel(n_jobs=-1)(
    delayed(process_one_claim_full)(
        row.verdict,
        row.statement,
        row.statement_date,
        row.statement_originator,
        posts_csv[
            (posts_csv["post_created"] >= row.statement_date) &
            (posts_csv["post_created"] <= row.statement_date + pd.Timedelta(days=days_window))
        ],
        top_k
    )
    # Get tqdm bar to visualise the process
    for row in tqdm(claims_csv.itertuples(index=False), total=len(claims_csv), desc="Matching claims (r2)")
)

# Flatten and merge results
flat_records_r2 = [rec for sub in results_r2 for rec in sub]
flat_df_r2 = pd.DataFrame(flat_records_r2)

# Join with all post variables
full_output_r2 = flat_df_r2.merge(posts_csv, on="post_id", how="left")
print(f"[r2] Pairs after merge: {len(full_output_r2)}")

# Dupe logic (just if needed)
# Deduplicate on claim + post, Logic Time Phase 1
'''
if not full_output_r2.empty:
    dupes_r2 = full_output_r2[full_output_r2.duplicated(subset=["claim_text", "post_id", "author_claim"], keep=False)]
    print("[r2] Duplicate claim–post pairs found:", len(dupes_r2))
    if not dupes_r2.empty:
        dupes_r2.to_csv("dupe_debug_r2.csv", index=False)
        print("[r2] Duplicates written to dupe_debug_r2.csv")
else:
    print("[r2] No pairs found; nothing saved.")
'''

# Save output to predefined output path
full_output_r2.to_csv(output_path_r2, index=False)
print(f"[r2] Saved to {output_path_r2} with {len(full_output_r2)} claim–post pairs.")


