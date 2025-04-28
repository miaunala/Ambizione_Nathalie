import pandas as pd
import os
from detoxify import Detoxify
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from collections import Counter

# sentiment
# abusive language
# misogyny

def preprocessing_comments(comments):
    flat_comments = comments[~(comments['comment_text'].isna() & comments['reply_text'].isna())].copy()

    flat_comments['text_for_analysis'] = flat_comments.apply(
        lambda row: row['reply_text'] if pd.notna(row.get('reply_id')) else row['comment_text'],
        axis=1
    )
    return flat_comments


def sentiment_model_analysis():
    model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)
    return sentiment_task


def toxicity_model_analysis():
    toxicity_task = Detoxify("multilingual")
    return toxicity_task


def misogyny_model_analysis():
    haz_model_misogyny = "annahaz/xlm-roberta-base-misogyny-sexism-indomain-mix-bal"
    misogyny_haz = pipeline("text-classification", model=haz_model_misogyny, tokenizer=haz_model_misogyny)
    return misogyny_haz


def split_into_chunks(text, tokenizer, max_tokens=512, stride=256):
    tokens = tokenizer.encode(text, truncation=False)

# this should never be evoked, bcs it should be beforehand
    if len(tokens) <= max_tokens:
        return [text]

    chunks = []
    for i in range(0, len(tokens), stride):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
        if i + max_tokens >= len(tokens):
            break

    return chunks


def aggregate_predictions(predictions):
    if not predictions:
        return {"label": "UNKNOWN", "score": 0.0}

    label_scores = {}
    for pred in predictions:
        label = pred["label"]
        label_scores[label] = label_scores.get(label, 0) + pred["score"]

    final_label = max(label_scores, key=label_scores.get)
    avg_score = label_scores[final_label] / len(predictions)

    return {"label": final_label, "score": avg_score}


def add_predictions_to_df(df, sentiment_results, toxicity_results_raw, misogyny_results):
    df["sentiment_label"] = [res["label"] for res in sentiment_results]
    df["sentiment_score"] = [res["score"] for res in sentiment_results]

    tox_keys = list(toxicity_results_raw[0].keys())
    tox_values = {key: [] for key in tox_keys}
    for batch_dict in toxicity_results_raw:
        for key in tox_keys:
            tox_values[key].extend(batch_dict[key])
    for key in tox_keys:
        df[f"toxicity_{key}"] = tox_values[key]

    df["misogyny_label"] = [res["label"] for res in misogyny_results]
    df["misogyny_score"] = [res["score"] for res in misogyny_results]

    return df

def batch_processing(df, batch_size=32):
    sentiment_model = sentiment_model_analysis()
    toxicity_model = toxicity_model_analysis()
    misogyny_model = misogyny_model_analysis()
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment")

    max_length = tokenizer.model_max_length  # typically 512

    texts = df['text_for_analysis'].tolist()

    sentiment_results = []
    toxicity_results_raw = []
    misogyny_results = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        # Sentiment Prediction
        for text in batch:
            try:
                tokens = tokenizer.encode(text, truncation=False)

                if len(tokens) <= max_length:
                    # Short enough, normal prediction
                    preds = sentiment_model(text, truncation=True)
                    sentiment_results.extend(preds)
                else:
                    # Too long, split into chunks
                    chunks = split_into_chunks(text, tokenizer, max_tokens=max_length, stride=max_length // 2)
                    chunk_preds = []
                    for chunk in chunks:
                        preds = sentiment_model(chunk, truncation=True)
                        chunk_preds.extend(preds)

                    final_pred = aggregate_predictions(chunk_preds)
                    sentiment_results.append(final_pred)

            except Exception as e:
                print(f"Error while processing sentiment:\n{text}\n{e}")
                sentiment_results.append({"label": "ERROR", "score": 0.0})

        # Toxicity Prediction
        try:
            toxicity_batch = toxicity_model.predict(batch)
            toxicity_results_raw.append(toxicity_batch)
        except Exception as e:
            print(f"Error while processing toxicity: {e}")
            toxicity_results_raw.append({
                "toxicity": [0.0] * len(batch),
                "severe_toxicity": [0.0] * len(batch),
                "obscene": [0.0] * len(batch),
                "identity_attack": [0.0] * len(batch),
                "insult": [0.0] * len(batch),
                "threat": [0.0] * len(batch),
                "sexual_explicit": [0.0] * len(batch)
            })

        # Misogyny Prediction
        try:
            ############ have to try, it depends on the pipeline
            misogyny_batch = misogyny_model(batch, truncation=False)
            #misogyny_batch = misogyny_model(batch, truncation=True)
            misogyny_results.extend(misogyny_batch)
        except Exception as e:
            print(f"Error while processing misogyny: {e}")
            misogyny_results.extend([{"label": "ERROR", "score": 0.0}] * len(batch))

    return add_predictions_to_df(df, sentiment_results, toxicity_results_raw, misogyny_results)

def main():
    source = r"/Users/nathalieguibert/Desktop/ResAss_Klüser_FS25/Ambizione Nathalie/Sexually Explicit Deepfakes/Code/"

    data_files = [
        #"flattened_comments_carahuntermla.csv",
        #"flattened_comments_angelaraynermp.csv",
        "flattened_comments_donorpool.csv"
    ]

    for data_file in data_files:
        print(f"Processing: {data_file}")
        flat_comments = pd.read_csv(os.path.join(source, data_file))

        flat_comments = preprocessing_comments(flat_comments)

        flat_comments = batch_processing(flat_comments, batch_size=32)

        flat_comments.to_csv(f"{data_file.replace('.csv', '')}_scores.csv", index=False)
        print(f"Saved to {data_file.replace('.csv', '')}_scores.csv")

if __name__ == "__main__":
    main()