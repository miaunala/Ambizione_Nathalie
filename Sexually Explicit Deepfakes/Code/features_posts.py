import pandas as pd
import os
from detoxify import Detoxify
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

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
    # old model
    #haz_model_misogyny = "annahaz/xlm-roberta-base-misogyny-sexism-indomain-mix-bal"
    #misogyny_haz = pipeline("text-classification", model=haz_model_misogyny, tokenizer=haz_model_misogyny)
    #return misogyny_haz

    model_path = "tum-nlp/bertweet-sexism"
    misogyny_task = pipeline("text-classification", model=model_path, tokenizer=model_path)
    return misogyny_task


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

    texts = df['text_for_analysis'].tolist()

    sentiment_results = []
    toxicity_results_raw = []
    misogyny_results = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        # Sentiment Prediction
        sentiment_batch = sentiment_model(batch, truncation=True, max_length=512)
        sentiment_results.extend(sentiment_batch)

        # Toxicity Prediction
        toxicity_batch = toxicity_model.predict(batch)
        toxicity_results_raw.append(toxicity_batch)

        # Misogyny Prediction
        misogyny_batch = misogyny_model(batch, truncation=True, max_length=512)
        misogyny_results.extend(misogyny_batch)

    return add_predictions_to_df(df, sentiment_results, toxicity_results_raw, misogyny_results)


def main():
    source = r"/Users/nathalieguibert/Desktop/ResAss_Klüser_FS25/Ambizione Nathalie/Sexually Explicit Deepfakes/Code/"

    data_files = [
        "flattened_comments_carahuntermla.csv",
        "flattened_comments_angelaraynermp.csv",
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