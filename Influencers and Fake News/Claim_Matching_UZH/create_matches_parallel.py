from datetime import timedelta
import pandas as pd
from joblib import Parallel, delayed, Memory
from tqdm import tqdm
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
import os

nltk.download("punkt")
nltk.download("punkt_tab")


# load files from before
# für local
base_path = "/Users/nathalieguibert/Desktop/ResAss_Klüser_FS25/Ambizione Nathalie/Influencers and Fake News/Claim_Matching_UZH/data/"
claims_csv = pd.read_csv(base_path + "analysis_claims_df.csv")
posts_csv = pd.read_csv(base_path + "analysis_posts_df.csv")

print(f"Initial claims count: {len(claims_csv)}")

# Drop full-row duplicates
claims_csv = claims_csv.drop_duplicates()
print(len(claims_csv))

# Drop duplicate statements by different authors
claims_csv = claims_csv.sort_values(by="statement_date", ascending=False)
claims_csv = claims_csv.drop_duplicates(subset="statement", keep="first")
print(len(claims_csv))

# drop duplciates where same post
print(f"After deduplicating claims count: {len(claims_csv)}")
print(f"Posts before filtering: {len(posts_csv)}")
posts_csv = posts_csv.drop_duplicates()
print(f"after filtering: {len(posts_csv)}")


# Convert to datetime
claims_csv["statement_date"] = pd.to_datetime(claims_csv["statement_date"])
posts_csv["post_created"] = pd.to_datetime(posts_csv["post_created"])


##### TOKEN SIMILARITY: BM25 ######
# Load pretrained Sentence Transformer Model
# Standard Model, auch genutzt im Claim Matching paper
# datum darf nicht mehr als 28 tage auseinander sein


# Filter claims within extended post date range
top_k = 1000
days_window = 28
min_post_date = posts_csv["post_created"].min() - pd.DateOffset(months=3)
max_post_date = posts_csv["post_created"].max() + pd.DateOffset(months=3)
claims_csv = claims_csv[
    (claims_csv["statement_date"] >= min_post_date) &
    (claims_csv["statement_date"] <= max_post_date)
].reset_index(drop=True)

# Assign post_id
posts_csv = posts_csv.reset_index(drop=True)
posts_csv["post_id"] = posts_csv.index

# Setup cache
memory = Memory("data/joblib_cache", verbose=1)


@memory.cache
def process_one_claim_full(verdict, claim_text, claim_date, author_claim, matching_posts_df, top_k):
    print(f"\nProcessing claim: {claim_text[:80]!r}")
    print(f"Claim date: {claim_date.strftime('%Y-%m-%d')}, Matching posts in timeframe: {len(matching_posts_df)}")

    tokenised_claim = [word.lower() for word in word_tokenize(claim_text)]

    tokenised_posts = []
    original_rows = []
    for _, row in matching_posts_df.iterrows():
        text = str(row["text_analysis"])
        tokens = [word.lower() for word in word_tokenize(text)]
        if tokens:
            tokenised_posts.append(tokens)
            original_rows.append(row)

    if not tokenised_posts:
        return []

    bm25 = BM25Okapi(tokenised_posts)
    scores = bm25.get_scores(tokenised_claim)

    if max(scores) == 0.000:
        return []

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    results = []
    seen_post_ids = set()

    for i in top_indices:
        post_id = int(original_rows[i]["post_id"])
        if post_id in seen_post_ids:
            continue
        seen_post_ids.add(post_id)

        results.append({
            "verdict": verdict,
            "claim_text": claim_text,
            "author_claim": author_claim,
            "post_text": original_rows[i]["text_analysis"],
            "post_id": post_id,
            "score": float(scores[i])
        })

    return results
    # worked bisher

# Both directions output + Remove old output
output_path_r1 = "data/data/flat_pairs_bm25_r1.csv"  # [claim_date-28d, claim_date]
output_path_r2 = "data/data/flat_pairs_bm25_r2.csv"  # [claim_date, claim_date+28d]

for _p in (output_path_r1, output_path_r2):
    if os.path.exists(_p):
        os.remove(_p)
        print(f"Old output file deleted: {_p}")


########## Direction r1 ##########
# Richtung 1: Claims - Posts
# Zeit 1: 28 Tage zuvor

# commented out because we already did
print("\n[r1] Matching posts in window: [claim_date - 28 days, claim_date]")
# Parallel processing
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
    for row in tqdm(claims_csv.itertuples(index=False), total=len(claims_csv), desc="Matching claims (r1)")
)

# Flatten and merge results
flat_records_r1 = [record for sublist in results_r1 for record in sublist]
flat_df_r1 = pd.DataFrame(flat_records_r1)

# Join with all post variables
full_output_r1 = flat_df_r1.merge(posts_csv, on="post_id", how="left")
print(len(full_output_r1))

# Deduplicate on claim + post, Logic Time Phase 1
'''
# needs to be commented out later
if not full_output_r1.empty:
    dupes_r1 = full_output_r1[full_output_r1.duplicated(subset=["claim_text", "post_id", "author_claim"], keep=False)]
    print("[r1] Duplicate claim–post pairs found:", len(dupes_r1))
    if not dupes_r1.empty:
        dupes_r1.to_csv("dupe_debug_r1.csv", index=False)
        print("[r1] Duplicates written to dupe_debug_r1.csv")
else:
    print("[r1] No pairs found; nothing saved.")
'''


full_output_r1.to_csv(output_path_r1, index=False)
print(f"[r1] Saved to {output_path_r1} with {len(full_output_r1)} claim–post pairs.")


#### TESTING ####
def window_count(row, direction, days=28):
    if direction == "r1":  # 28 days BEFORE up to claim date (inclusive)
        m = (posts_csv["post_created"] >= row.statement_date - pd.Timedelta(days=days)) & \
            (posts_csv["post_created"] <= row.statement_date)
    else:  # r2: claim date up to 28 days AFTER (inclusive)
        m = (posts_csv["post_created"] >= row.statement_date) & \
            (posts_csv["post_created"] <= row.statement_date + pd.Timedelta(days=days))
    return int(m.sum())



#####


########## Direction r2 ##########
# Richtung 2: Claims - Posts
# # Zeit 2: 28 Tage danach
print("\n[r2] Matching posts in window: [claim_date, claim_date + 28 days]")
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
    for row in tqdm(claims_csv.itertuples(index=False), total=len(claims_csv), desc="Matching claims (r2)")
)

flat_records_r2 = [rec for sub in results_r2 for rec in sub]
flat_df_r2 = pd.DataFrame(flat_records_r2)
full_output_r2 = flat_df_r2.merge(posts_csv, on="post_id", how="left") if not flat_df_r2.empty else pd.DataFrame()
print(f"[r2] Pairs after merge: {len(full_output_r2)}")

# Dupe logic (just if needed)
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

full_output_r2.to_csv(output_path_r2, index=False)
print(f"[r2] Saved to {output_path_r2} with {len(full_output_r2)} claim–post pairs.")


