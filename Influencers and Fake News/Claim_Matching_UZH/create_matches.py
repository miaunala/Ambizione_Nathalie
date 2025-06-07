import os
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import openai
import random
import time
import csv
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
import nltk
nltk.download("punkt")

# load files from before
base_path = "/Users/nathalieguibert/Desktop/ResAss_Klüser_FS25/Ambizione Nathalie/Influencers and Fake News/Claim_Matching_UZH/data/"
claims =...
posts = ...


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

# maybe this logic in main()??
#create matches claims <- posts
for every in claims:
    ...


# create matches posts <- claims




# only later choose accoutns on accounts csv??