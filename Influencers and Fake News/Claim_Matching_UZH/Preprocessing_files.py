import os
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import openai
import random
import time
import csv

base_path = "/Users/nathalieguibert/Desktop/ResAss_Klüser_FS25/Ambizione Nathalie/Influencers and Fake News/Claim_Matching_UZH/data/"
# File paths
accounts_file = "account_clusters.csv"
posts_file = "ct_combined_20-23.csv"
claims_file = "politifact_facteck_data.csv"
output_dir = "openai_train_json"

# test it if we already just do it as directory
#os.makedirs(output_dir, exist_ok=True)


# Load the data
# braucht es das überhaupt??
accounts_df = pd.read_csv(base_path + accounts_file)
posts_df = pd.read_csv(base_path + posts_file)
claims_df = pd.read_csv(base_path + claims_file)

# Only keep relevant columns on claims
print(claims_df.columns.tolist())
claims_df = claims_df[claims_df["verdict"].isin(["false", "mostly-false"])]
# oder variable false  == TRUE
# check type beforehand
# check whether alls false and mostly-false are in category false == True
claims_df = claims_df[claims_df["false"] == "True"]

claims_df = claims_df[["verdict", "statement_originator", "statement", "statement_date", "factcheck_date"]] # vielelicht wegmachen und erst später nach analyse
claims_df.to_csv("analysis_claims_df.csv", index=False)

# only keep english posts
posts_df = posts_df[posts_df["lang"] == "en"]



posts_df["description"] = posts_df["description"].fillna("")
posts_df["media_text"] = posts_df["media_text"].fillna("")
posts_df["text_analysis"] = posts_df["description"] + " " + posts_df["media_text"]

posts_df.to_csv("analysis_posts_df.csv", index=False)
print("joa ")
