import pandas as pd
import os
from transformers import pipeline, AutoTokenizer
from detoxify import Detoxify

# sentiment
# abusive language
# misogyny




source = r"/Users/nathalieguibert/Desktop/ResAss_Klüser_FS25/Ambizione Nathalie/Sexually Explicit Deepfakes/Code/"

data_file = "flattened_comments_carahuntermla.csv" #"flattened_comments_carahuntermla.csv" # flattened_comments_angelaraynermp.csv

flat_comments = pd.read_csv(os.path.join(source, data_file))

print(flat_comments.head())

# Vorverarbeitung: Entferne Zeilen ohne jeglichen Text
flat_comments = flat_comments[~(flat_comments['comment_text'].isna() & flat_comments['reply_text'].isna())]

# 1. Wähle den passenden Text: reply oder comment
flat_comments['text_for_analysis'] = flat_comments.apply(
    lambda row: row['reply_text'] if pd.notna(row.get('reply_id')) else row['comment_text'],
    axis=1
)

# 2. Alles in Strings umwandeln und fehlende Werte mit "" ersetzen
flat_comments['text_for_analysis'] = flat_comments['text_for_analysis'].fillna("").astype(str)

###### SENTIMENT ########
model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)

#### Toxicity ########
toxicity_model = Detoxify("multilingual")


# 3. Batchweise Sentiment-Analyse
texts = flat_comments['text_for_analysis'].tolist()
batch_size = 32
sentiment_results = []
toxicity_results = []

for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    # Sentiment
    batch_sentiment = sentiment_task(batch)
    sentiment_results.extend(batch_sentiment)
    # Abusive language
    batch_toxicity = toxicity_model.predict(batch)
    toxicity_results.append(batch_toxicity)

    # Detoxify-Ergebnisse formatieren
tox_keys = list(toxicity_results[0].keys())
tox_values = {key: [] for key in tox_keys}

for batch_dict in toxicity_results:
    for key in tox_keys:
        tox_values[key].extend(batch_dict[key])
#


# 4. Ergebnisse zurück in den DataFrame
flat_comments['sentiment_label'] = [res['label'] for res in sentiment_results]
flat_comments['sentiment_score'] = [res['score'] for res in sentiment_results]

# Schreibe Detoxify-Toxicity-Resultate ins DataFrame
for key in tox_keys:
    flat_comments[f"toxicity_{key}"] = tox_values[key]


print(flat_comments)

###### SAVING SCORES #####3
flat_comments.to_csv(f"{data_file.replace('.csv', '')}_scores.csv", index=False)
### applied to the samples
