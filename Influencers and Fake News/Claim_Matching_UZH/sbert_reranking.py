from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

# Setting base path, loading SBert Model, setting chunk size for cosine similarity
base_path = "/Users/nathalieguibert/Desktop/ResAss_Klüser_FS25/Ambizione Nathalie/Influencers and Fake News/Claim_Matching_UZH/data/"
'''
# for cluster
base_path_cluster="data/"
'''
model = SentenceTransformer("all-MiniLM-L6-v2")

COS_SIM_CHUNK = 500000

##### SEMANTIC SIMILARITY: SBert ######

# For both directions
for direction in ("r1", "r2"):

    # Based on direction build input and output path
    bm25_path = os.path.join(base_path, f"data/flat_pairs_bm25_{direction}.csv")
    out_path  = os.path.join(base_path, f"data/matches_sbert_{direction}.csv")
    print(f"\n[SBERT] Direction: {direction}")

    # Load input file
    inter_df = pd.read_csv(bm25_path, low_memory=False)

    # Extract only unique Claims and Posts to avoid multiple encoding, thus, saving time
    unique_claims = pd.Index(pd.unique(inter_df["claim_text"]))
    unique_posts  = pd.Index(pd.unique(inter_df["post_text"]))

    # Encoding unique claims with batch size of 256,
    # Converting results to NumPy,
    # Normalising embeddings with length,
    # such that dot product is only necessary and not need cosine similarity (dot product + normalised vectors)
    claim_vecs = model.encode(
        unique_claims.tolist(),
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32, copy=False)

    # Encoding unique posts with batch size of 256,
    # Converting results to NumPy,
    # Normalising embeddings with length,
    post_vecs = model.encode(
        unique_posts.tolist(),
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32, copy=False)

    ### Map each claim/post text to its corresponding vector index for efficient similarity lookup
    # Create a mapping from each unique claim text to its index in the claim_vecs array
    c_map = {t: i for i, t in enumerate(unique_claims)}
    # Create a mapping from each unique post text to its index in the post_vecs array
    p_map = {t: i for i, t in enumerate(unique_posts)}
    # Map each claim_text in the dataframe to its corresponding index in claim_vecs
    c_idx = inter_df["claim_text"].map(c_map).to_numpy()
    # # Map each post_text in the dataframe to its corresponding index in post_vecs
    p_idx = inter_df["post_text"].map(p_map).to_numpy()


    ### Cosine similarity (dot product of normalized vectors) via chunk processing
    # Allocate an empty array to store cosine similarity results
    sims = np.empty(len(inter_df), dtype=np.float32)
    # Compute cosine similarity in chunks to avoid memory overload
    # Cosine = dot product of normalised vectors (SBERT vectors are L2-normalised)
    for start in tqdm(range(0, len(inter_df), COS_SIM_CHUNK),
                      desc=f"Cosine (chunks) {direction}", unit="chunk"):
        end = min(start + COS_SIM_CHUNK, len(inter_df))
        # Compute dot product between claim and post embeddings for current chunk
        sims[start:end] = (claim_vecs[c_idx[start:end]] * post_vecs[p_idx[start:end]]).sum(axis=1)
    # Assign cosine similarity scores to DataFrame column
    inter_df["cos_sim"] = sims

    # Per-claim reranking of Posts (tie-break by BM25 score if Cosine Similarity same)
    if "score" in inter_df.columns:
        inter_df = inter_df.sort_values(["claim_text", "cos_sim", "score"],
                                        ascending=[True, False, False])
    else:
        inter_df = inter_df.sort_values(["claim_text", "cos_sim"],
                                        ascending=[True, False])

    # Saving logic
    inter_df.to_csv(out_path, index=False)
    print(f"Results saved -> {out_path} ({len(inter_df):,} rows)")



