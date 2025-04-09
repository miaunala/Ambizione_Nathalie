import json
import pandas as pd
import os

def load_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def flatten_comments(data):
    flattened_comments = []

    for post in data:
        post_id = post.get('_id')
        post_timestamp = post.get('created_at', {}).get('$date')

        for comment in post.get('comments', []):
            flat_comment = {
                "post_id": post_id,
                "post_timestamp": post_timestamp,
                "comment_id": comment.get('_id'),
                "comment_timestamp": comment.get('created_at', {}).get('$date'),
                "comment_username": comment.get('username'),
                "comment_text": comment.get('text'),
                "comment_scrape_date": comment.get('timestamp', {}).get('$date'),
                "comment_user_id": comment.get('user_id'),
                "comment_like_count": comment.get('like_count'),
                "comment_reply_count": comment.get('reply_count'),
                "comment_did_report_as_spam": comment.get('did_report_as_spam'),
                "comment_hashtags": comment.get('hashtags'),
                "comment_mentions": comment.get('mentions'),
            }

            flattened_comments.append(flat_comment)

            if flat_comment["comment_reply_count"]:
                for reply in comment.get('replies', []):
                    flat_reply = {
                        "post_id": post_id,
                        "post_timestamp": post_timestamp,
                        "comment_id": flat_comment["comment_id"],
                        "comment_username": flat_comment["comment_username"],
                        "reply_id": reply.get('_id'),
                        "reply_timestamp": reply.get('created_at', {}).get('$date'),
                        "reply_username": reply.get('username'),
                        "reply_text": reply.get('text'),
                        "reply_scrape_date": reply.get('timestamp', {}).get('$date'),
                        "reply_user_id": reply.get('user_id'),
                        "reply_like_count": reply.get('like_count'),
                        "reply_hashtags": reply.get('hashtags'),
                        "reply_mentions": reply.get('mentions', []),
                    }
                    flattened_comments.append(flat_reply)

    return pd.DataFrame(flattened_comments)

def deduplicate_comments(df, is_hunter=False):
    if not is_hunter:
        return df

    # Create a unified scrape date
    df['scrape_date'] = df.apply(
        lambda row: row['reply_scrape_date'] if pd.notna(row.get('reply_id')) else row['comment_scrape_date'],
        axis=1
    )

    df['scrape_date'] = pd.to_datetime(df['scrape_date'], errors='coerce')

    # Split into replies and comments
    is_reply = pd.notna(df['reply_id'])
    df_replies = df[is_reply].drop_duplicates(subset='reply_id', keep='first')
    df_comments_only = df[~is_reply].drop_duplicates(subset='comment_id', keep='first')

    # Combine again
    df_deduped = pd.concat([df_comments_only, df_replies], ignore_index=True)
    df_deduped.sort_values(by=['post_id', 'scrape_date'], inplace=True)

    return df_deduped

def export_to_csv(df, output_name):
    df.to_csv(output_name, index=False)
    print(f"Exported to {output_name}")

def main():
    source = "/Users/nathalieguibert/Desktop/ResAss_Klüser_FS25/Ambizione Nathalie/Sexually Explicit Deepfakes/"
    data_file = "Posts_angelaraynermp.json"  # change this to data_hunter_mla if needed
    file_path = os.path.join(source, data_file)

    is_hunter = data_file == "Posts_carahuntermla.json"

    data = load_json_file(file_path)
    df_comments = flatten_comments(data)
    df_comments = deduplicate_comments(df_comments, is_hunter=is_hunter)

    print(df_comments.head())
    print(df_comments.shape)
    export_to_csv(df_comments, f"flattened_comments_{data_file.replace('Posts_', '').replace('.json', '')}.csv")

if __name__ == "__main__":
    main()
