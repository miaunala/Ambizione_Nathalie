from datetime import timedelta
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
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
'''

pairs_r1_z1 = []
top_posts_per_claim = {}

flat_pairs = []

for claim_row in tqdm(claims_csv.itertuples(index=False), total=len(claims_csv), desc="Processing claims"):
    claim_text = claim_row.statement
    claim_date = claim_row.statement_date

    start_date_claim = claim_date - timedelta(days=28)
    end_date_claim = claim_date

    tokenised_claim = nlp(claim_text)

    #changes with every claim
    matching_timeframe_posts = posts_csv[
        (posts_csv["post_created"] >= start_date_claim) &
        (posts_csv["post_created"] <= end_date_claim)
        ]

    tokenised_posts = []
    original_post_texts = []

    # just does the pairs together
    for post_row in matching_timeframe_posts.itertuples(index=False):
        # format into string if post consists (only) of non-alphabetical stuff
        post_text = str(post_row.text_analysis)
        original_post_texts.append(post_text)
        tokenised_post = nlp(post_text)
        tokenised_posts.append(tokenised_post)

    if not tokenised_posts:
        continue

    # Build BM25 corpus
    bm25 = BM25Okapi(tokenised_posts)

    # Score claim against all posts that fall within a timeframe
    scores = bm25.get_scores(tokenised_claim)
    print(scores)
    # how to proceed if all scores are 0.000

    ### Maybe not necessary, only for development ###
    for post_text, score in zip(original_post_texts, scores):
        pairs_r1_z1.append((claim_text, post_text, score))

        # reset after every claim
    print(len(pairs_r1_z1), "endgültige Länge")


    #print(scores)
    if max(scores) == float(0.0):
        print("no match")
        continue

    ### STILL NEEDS TO BE TESTED IF LAST LOOP IS THROUGH?? ###
    # would be great argument in function
    top_k = 1000

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    print(top_indices)

    for i in top_indices:
        flat_pairs.append((claim_text, original_post_texts[i], scores[i]))

    # Ergebnis abspeichern
    #top_posts_per_claim[claim_text] = top_results  # oder claim_text als Schlüssel
    #print(top_posts_per_claim[claim_text])

flat_pairs = pd.DataFrame(flat_pairs, columns=["claim_text", "post_text", "score"])
flat_pairs.to_csv("data/data/flat_pairs.csv", index=False)


#checken ob ältere claims überhaupt drinnen sind (weil zu alt und keine posts) und die dann rauslöschen

print(pairs_r1_z1)
print("ahjo")'''

''' Pseudocode 

for claim_row in pairs_r1_z1:
    claim_row[score]
pairs_r1_z1[score]

- temporary list for claims
temp_claim = []
- scores accessen
for pair in pairs_r1_z1:
    
    
'''







### neue Versie 1

'''
# SBERT Model laden (für später)
model = SentenceTransformer("all-MiniLM-L6-v2")

def process_one_claim(claim_row, posts_df, top_k=1000):
    nlp = spacy.load("en_core_web_sm")
    claim_text = claim_row.statement
    claim_date = claim_row.statement_date
    start_date = claim_date - timedelta(days=28)
    end_date = claim_date

    # Filter posts im Zeitfenster
    matching_posts = posts_df[
        (posts_df["post_created"] >= start_date) &
        (posts_df["post_created"] <= end_date)
    ]

    if matching_posts.empty:
        return []

    # Tokenisiere Claim
    tokenised_claim = [token.text for token in nlp(claim_text) if token.is_alpha]

    # Tokenisiere alle Posts
    tokenised_posts = []
    original_texts = []
    for text in matching_posts["text_analysis"]:
        text = str(text)
        tokens = [token.text for token in nlp(text) if token.is_alpha]
        if tokens:
            tokenised_posts.append(tokens)
            original_texts.append(text)

    if not tokenised_posts:
        return []

    # BM25 Scores berechnen
    bm25 = BM25Okapi(tokenised_posts)
    scores = bm25.get_scores(tokenised_claim)

    if max(scores) == 0.0:
        return []

    # Top K auswählen
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    del nlp
    return [(claim_text, original_texts[i], scores[i]) for i in top_indices]

# Parallele Verarbeitung
results = Parallel(n_jobs=-1)(
    delayed(process_one_claim)(row, posts_csv)
    for row in tqdm(claims_csv.itertuples(index=False), total=len(claims_csv))
)

# Flach machen & speichern
flat_pairs = [pair for sublist in results for pair in sublist]
flat_df = pd.DataFrame(flat_pairs, columns=["claim_text", "post_text", "score"])
flat_df.to_csv("data/data/flat_pairs.csv", index=False)

'''
#neue Versie 2

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

# example from gpt
'''from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
import nltk
nltk.download("punkt")

# Token similarity: BM25
def bm25_topk(claim, tweets, k=1000):
    tokenized_corpus = [nltk.word_tokenize(doc.lower()) for doc in tweets]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_claim = nltk.word_tokenize(claim.lower())
    scores = bm25.get_scores(tokenized_claim)
    topk_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return topk_idx

# Semantic similarity: Sentence-BERT
model = SentenceTransformer('all-MiniLM-L6-v2')

def sbert_rerank(claim, candidate_tweets):
    embeddings = model.encode([claim] + candidate_tweets, convert_to_tensor=True)
    sim_scores = util.cos_sim(embeddings[0], embeddings[1:])[0]
    ranked_indices = sim_scores.argsort(descending=True)
    return [candidate_tweets[i] for i in ranked_indices]'''

# choose tweet-claim pairs top 10?




# create matches posts <- claims




# only later choose accoutns on accounts csv??
