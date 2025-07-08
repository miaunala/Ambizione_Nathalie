from datetime import timedelta
import pandas as pd
from joblib import Parallel, delayed, Memory
from tqdm import tqdm
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize

nltk.download("punkt")
nltk.download("punkt_tab")




# load files from before
# für local
base_path = "/Users/nathalieguibert/Desktop/ResAss_Klüser_FS25/Ambizione Nathalie/Influencers and Fake News/Claim_Matching_UZH/data/"
claims_csv = pd.read_csv(base_path + "analysis_claims_df.csv")
posts_csv = pd.read_csv(base_path + "analysis_posts_df.csv")

#claims_csv = pd.read_csv("data/data/analysis_claims_df.csv")
#posts_csv = pd.read_csv("data/data/analysis_posts_df.csv")

# Convert to datetime
claims_csv["statement_date"] = pd.to_datetime(claims_csv["statement_date"])
posts_csv["post_created"] = pd.to_datetime(posts_csv["post_created"])


# sentence splitter for posts

##### TOKEN SIMILARITY: BM25 ######
# Load pretrained Sentence Transformer Model
#### Frage: pretrained model benutzen oder noch andere finetuned models? (gibt 10,000 andere pretrained models) oder selbst finetunen ggf?
# Standard Model, auch genutzt im Claim Matching paper
# datum darf nicht mehr als 28 tage auseinander sein

# Richtung 1: Claims - Posts
# Zeit 1: 28 Tage zuvor
############################ alte versie ##############

# funktionierende Version 1

'''
# Main processing function
# Parameters
top_k = 1000
days_window = 28

def process_one_claim(claim_row, posts_df):
    claim_text = claim_row.statement
    claim_date = claim_row.statement_date
    start_date = claim_date - pd.Timedelta(days=days_window)
    end_date = claim_date

    # Filter posts within time window
    matching_posts = posts_df[
        (posts_df["post_created"] >= start_date) &
        (posts_df["post_created"] <= end_date)
    ]

    # Tokenise claim
    tokenised_claim = [word.lower() for word in word_tokenize(claim_text) if word.isalpha()]

    # Tokenise posts
    tokenised_posts = []
    original_texts = []
    for text in matching_posts["text_analysis"].astype(str):
        tokens = [word.lower() for word in word_tokenize(text) if word.isalpha()]
        if tokens:
            tokenised_posts.append(tokens)
            original_texts.append(text)

    if not tokenised_posts:
        return []

    # BM25 scoring
    bm25 = BM25Okapi(tokenised_posts)
    scores = bm25.get_scores(tokenised_claim)

    print(scores)

    if max(scores) == 0.0:
        return []

    # Select top K results
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [(claim_text, original_texts[i], scores[i]) for i in top_indices]

# Parallel processing
results = Parallel(n_jobs=-1)(
    delayed(process_one_claim)(row, posts_csv)
    for row in tqdm(claims_csv.itertuples(index=False), total=len(claims_csv))
)

# Flatten and save
flat_pairs = [pair for sublist in results for pair in sublist]
flat_df = pd.DataFrame(flat_pairs, columns=["claim_text", "post_text", "score"])
flat_df.to_csv("data/data/flat_pairs.csv", index=False) '''




# V3ersie 2 elaborierter Version
# Parameters
top_k = 1000
days_window = 28

# Joblib memory cache
memory = Memory("data/joblib_cache", verbose=1)

# Cached claim–post matcher
@memory.cache
def process_one_claim_cached(claim_text, claim_date, post_texts, top_k=1000):
    print(f"\nProcessing claim: {claim_text[:80]!r}")
    print(f"Claim date: {claim_date.strftime('%Y-%m-%d')}, Matching posts in timeframe: {len(post_texts)}")

    tokenised_claim = [word.lower() for word in word_tokenize(claim_text)]

    tokenised_posts = []
    original_texts = []
    for text in post_texts:
        tokens = [word.lower() for word in word_tokenize(str(text))]
        if tokens:
            tokenised_posts.append(tokens)
            original_texts.append(text)

    if not tokenised_posts:
        return []

    bm25 = BM25Okapi(tokenised_posts)
    scores = bm25.get_scores(tokenised_claim)

    if max(scores) == 0.000:
        return []

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [(claim_text, original_texts[i], float(scores[i])) for i in top_indices]

# Run matching in parallel
results = Parallel(n_jobs=-1)(
    delayed(process_one_claim_cached)(
        row.statement,
        row.statement_date,
        posts_csv[
            (posts_csv["post_created"] >= row.statement_date - pd.Timedelta(days=days_window)) &
            (posts_csv["post_created"] <= row.statement_date)
        ]["text_analysis"].tolist(),
        top_k
    )
    for row in tqdm(claims_csv.itertuples(index=False), total=len(claims_csv))
)

# Flatten and save results
flat_pairs = [pair for sublist in results for pair in sublist]
flat_df = pd.DataFrame(flat_pairs, columns=["claim_text", "post_text", "score"])
flat_df.to_csv("data/data/flat_pairs.csv", index=False)


#SBert
'''
embeddings = []
for pairs in pairs_r1_z1:
    embeddings = model.encode(pairs)

#Similarities
for embedding in embeddings:
    similarity_r1_z1 = model.similarity(embeddings, embeddings)'''

# Zeit 2: 28 Tage danach



# also ist [claim, post]



##### Reranking ######



##### SEMANTIC SIMILARITY: SBert #####

# define possible post/claim (for both directions) for each post/claim
#

# create function for bm25 that can be called in both directions

# create function for Sbert that can be called in both directions
# maybe already incldues reranking?? or as own function?

