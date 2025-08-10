from datetime import timedelta
import pandas as pd
from joblib import Parallel, delayed, Memory
from tqdm import tqdm
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.util import cos_sim
import os
import numpy as np
import torch



nltk.download("punkt")
nltk.download("punkt_tab")


# load files from before
# für local
base_path = "/Users/nathalieguibert/Desktop/ResAss_Klüser_FS25/Ambizione Nathalie/Influencers and Fake News/Claim_Matching_UZH/data/"
'''claims_csv = pd.read_csv(base_path + "analysis_claims_df.csv")
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

# Richtung 1: Claims - Posts
# Zeit 1: 28 Tage zuvor

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

# umbenennen in process_one_claim_bm25
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

# Remove old output
# muss das wirklich data/data/ sein (zumindest für local?)
output_path = "data/data/flat_pairs_bm25.csv"
if os.path.exists(output_path):
    os.remove(output_path)
    print("Old output file deleted.")

# Parallel processing
results = Parallel(n_jobs=-1)(
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
    for row in tqdm(claims_csv.itertuples(index=False), total=len(claims_csv), desc="Matching claims")
)

# Flatten and merge results
flat_records = [record for sublist in results for record in sublist]
flat_df = pd.DataFrame(flat_records)

# Join with all post variables
full_output = flat_df.merge(posts_csv, on="post_id", how="left")
print(len(full_output))

# Deduplicate on claim + post
# Check for duplicates before dropping

#### BRAUCHEN WIR DAS NOCH???
dupes = full_output[full_output.duplicated(subset=["claim_text", "post_id", "author_claim"], keep=False)]
print("Duplicate claim–post pairs found:", len(dupes))
if not dupes.empty:
    dupes.to_csv("dupe_debug.csv", index=False)
    print("Duplicates written to dupe_debug.csv")

#full_output.drop_duplicates(subset=["claim_text", "post_id", "author_claim"], inplace=True)
print(len(full_output))
#########

# Save intermediate step after BM25
full_output.to_csv(output_path, index=False)
print(f"Saved to {output_path} with {len(full_output)} unique claim–post pairs.")

'''
###### Step 2
inter_df = pd.read_csv(base_path+"data/"+"flat_pairs_bm25.csv")

print(f"Length df after flat BM25: {len(inter_df)}")

##### 29.7. Next Steps
'''
- schauen ob man die tokenisation rüberbringt von BM25 (wenns sein muss mit den embeddings)
- embeddings berechnen
- (cosine) similarity mit sbert
- beide Zeiten machen'''

# funktioniert aber  langsam##
#SBert
model = SentenceTransformer("all-MiniLM-L6-v2")

print(len(inter_df))
# Ensure required columns
if "claim_text" not in inter_df.columns or "post_text" not in inter_df.columns:
    raise ValueError("inter_df must contain 'claim_text' and 'post_text' columns")
print(len(inter_df))
# Drop NaNs to avoid encode errors
inter_df = inter_df.dropna(subset=["claim_text", "post_text"]).copy()
print(len(inter_df))
# 1) Build unique strings (so we encode each text only once)
unique_claims = pd.Index(pd.unique(inter_df["claim_text"]))
print(unique_claims)
unique_posts  = pd.Index(pd.unique(inter_df["post_text"]))
print(unique_posts)

# 2) Encode once per side (normalise so cosine == dot product)
claim_vecs = model.encode(
    unique_claims.tolist(),
    batch_size=256,               # adjust for your machine
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
)
post_vecs = model.encode(
    unique_posts.tolist(),
    batch_size=256,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
)

# 3) Map each row to the index of its precomputed embedding
c_map = {t: i for i, t in enumerate(unique_claims)}
print(c_map)
p_map = {t: i for i, t in enumerate(unique_posts)}
print(p_map)
c_idx = inter_df["claim_text"].map(c_map).to_numpy()
print(c_idx)
p_idx = inter_df["post_text"].map(p_map).to_numpy()
print(p_idx)

# 4) Vectorised cosine similarity (dot of normalised vectors)
inter_df["cos_sim"] = (claim_vecs[c_idx] * post_vecs[p_idx]).sum(axis=1)
print(inter_df["cos_sim"])

# 5) Save result
inter_df.to_csv("matches_sbert.csv", index=False)
print("end")
'''sims = []
for pair in tqdm(inter_df.itertuples(index=False), total=len(inter_df), desc="SBERT cosine", unit="pair"):
    claim_text = pair.claim_text
    post_text = pair.post_text

    c_emb = model.encode(claim_text)
    #print("Embedding claim:", c_emb)
    p_emb = model.encode(post_text)
    #print("Embedding post:", p_emb)

    sim_val = float(util.cos_sim(c_emb, p_emb))
    sims.append(sim_val)

    print("Cosine similarity:", sim_val)
    #print("end")

inter_df["cos_sim"] = sims

print("end")'''


# cosine similarity between SBert embeddings

'''
embeddings = []
for pairs in pairs_r1_z1:
    embeddings = model.encode(pairs)

#Similarities
for embedding in embeddings:
    similarity_r1_z1 = model.similarity(embeddings, embeddings)'''

# Zeit 2: 28 Tage danach
'''claims_csv = claims_csv[
    (claims_csv["statement_date"] >= min_post_date) &
    (claims_csv["statement_date"] <= max_post_date)
].reset_index(drop=True)'''


# also ist [claim, post]



##### Reranking ######



##### SEMANTIC SIMILARITY: SBert #####

# define possible post/claim (for both directions) for each post/claim
#

# create function for bm25 that can be called in both directions

# create function for Sbert that can be called in both directions
# maybe already incldues reranking?? or as own function?

