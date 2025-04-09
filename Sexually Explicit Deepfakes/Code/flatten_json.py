import json
import pandas as pd

source = r"/Users/nathalieguibert/Desktop/ResAss_Klüser_FS25/Ambizione Nathalie/Sexually Explicit Deepfakes/"

data_hunter_mla = r"Posts_carahuntermla.json"


data_rayner_mp = r"Posts_angelaraynermp.json"


# erstmal durch alle files schauen und dann modular machen
#def load_json_file(json_file):
    #with open(json_file, "r", encoding="utf-8") as f:
        #data = json.load(f)
        #return data


# Load the JSON file
# first round: data_hunter_mla in zweiter
file_path = f"{source}{data_rayner_mp}"
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print(data)

print("---------------1--------------")

# Flatten JSON at the comment level
flattened_comments = []

for post in data:
    print(json.dumps(post, indent=2, ensure_ascii=False))  # ensure_ascii=False keeps Unicode characters readable
    post_id = post.get('_id')
    print(post_id)
    post_timestamp = post.get('created_at', {}).get('$date')
    print(post_timestamp)

    print(".-------------------------")
    for comment in post.get('comments', []):
        print(comment)
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

        print(flat_comment)
        flattened_comments.append(flat_comment)
        print(flattened_comments)
        print(f"Comment done (Comment ID: {flat_comment['comment_id']})")
    print(f"Post Done (Post ID: {post_id})")


# Convert to DataFrame
df_comments = pd.DataFrame(flattened_comments)
print(df_comments)

# leere texte in comment_text or reply_text aus df löschen (bei allen9

'''
### HUNTER STUFF

# Create a unified scrape date
df_comments['scrape_date'] = df_comments.apply(
    lambda row: row['reply_scrape_date'] if pd.notna(row['reply_id']) else row['comment_scrape_date'],
    axis=1
)
print(df_comments)
print(df_comments["scrape_date"])

# Convert scrape_date to datetime
df_comments['scrape_date'] = pd.to_datetime(df_comments['scrape_date'])
print(df_comments["scrape_date"])

# Sort by most recent scrape date
df_comments.sort_values(by='scrape_date', ascending=False, inplace=True)
print(df_comments)

# Split replies and comments
# does this return ids??? -> evaluate pd.notna ...
is_reply = pd.notna(df_comments['reply_id'])
print(is_reply)
df_replies = df_comments[is_reply].drop_duplicates(subset='reply_id', keep='first')
print(df_replies)
df_comments_only = df_comments[~is_reply].drop_duplicates(subset='comment_id', keep='first')
print(df_comments_only)

# Combine again
df_comments_deduped = pd.concat([df_comments_only, df_replies], ignore_index=True)
print(df_comments_deduped)

# Final sorting
df_comments_deduped.sort_values(by=['post_id', 'scrape_date'], inplace=True)
print(df_comments_deduped)



# Replace original
df_comments = df_comments_deduped
print(df_comments)
'''


# einfach in die main zum schauen
# View or export
print(df_comments.head())
print(df_comments.shape)
print(df_comments)
print("ende")


# argument shadowing -> maybe needs to be fixed
#def df_to_csv(df_comments, name):
# irgendwie das flexibler mit dem name gestalten, dass gerade die namen, benutzt werden siehe argumente in funktionen definition
df_comments.to_csv(f"flattened_comments_raynermp.csv", index=False)


#def main():
