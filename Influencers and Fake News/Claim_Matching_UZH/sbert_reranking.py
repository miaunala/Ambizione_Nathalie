from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

base_path = "/Users/nathalieguibert/Desktop/ResAss_Klüser_FS25/Ambizione Nathalie/Influencers and Fake News/Claim_Matching_UZH/data/"
model = SentenceTransformer("all-MiniLM-L6-v2")

COS_SIM_CHUNK = 500000

###### Step 2
for direction in ("r1", "r2"):
    bm25_path = os.path.join(base_path, f"data/flat_pairs_bm25_{direction}.csv")
    out_path  = os.path.join(base_path, f"data/matches_sbert_{direction}.csv")
    print(f"\n[SBERT] Direction: {direction}")

    # Load
    inter_df = pd.read_csv(bm25_path, low_memory=False)

    # Encode unique texts once (normalized so cosine == dot product)
    unique_claims = pd.Index(pd.unique(inter_df["claim_text"]))
    unique_posts  = pd.Index(pd.unique(inter_df["post_text"]))

    claim_vecs = model.encode(
        unique_claims.tolist(),
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32, copy=False)

    post_vecs = model.encode(
        unique_posts.tolist(),
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32, copy=False)

    # Map rows to precomputed vectors
    c_map = {t: i for i, t in enumerate(unique_claims)}
    p_map = {t: i for i, t in enumerate(unique_posts)}
    c_idx = inter_df["claim_text"].map(c_map).to_numpy()
    p_idx = inter_df["post_text"].map(p_map).to_numpy()

    # Cosine similarity (dot product of normalized vectors)
    ### funtktioniert aber machen wir anders
    #inter_df["cos_sim"] = (claim_vecs[c_idx] * post_vecs[p_idx]).sum(axis=1)
    sims = np.empty(len(inter_df), dtype=np.float32)
    for start in tqdm(range(0, len(inter_df), COS_SIM_CHUNK),
                      desc=f"Cosine (chunks) {direction}", unit="chunk"):
        end = min(start + COS_SIM_CHUNK, len(inter_df))
        sims[start:end] = (claim_vecs[c_idx[start:end]] * post_vecs[p_idx[start:end]]).sum(axis=1)
    inter_df["cos_sim"] = sims

    # Per-claim reranking (tie-break by BM25 score if present)
    if "score" in inter_df.columns:
        inter_df = inter_df.sort_values(["claim_text", "cos_sim", "score"],
                                        ascending=[True, False, False])
    else:
        inter_df = inter_df.sort_values(["claim_text", "cos_sim"],
                                        ascending=[True, False])

    # Save
    inter_df.to_csv(out_path, index=False)
    print(f"Results saved -> {out_path} ({len(inter_df):,} rows)")


### old version, works alright
'''time_frames = ["r1", "r2"]
inter_df = pd.read_csv(base_path+"data/"+"flat_pairs_bm25_.csv")

print(f"Length df after flat BM25: {len(inter_df)}")



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


# 5) Reorder Values before saving result
inter_df = inter_df.sort_values(by=["claim_text", "cos_sim"], ascending=[True, False])
inter_df.to_csv("matches_sbert.csv", index=False)
print("end")'''