library("dplyr")
library("jsonlite")
library(readr)

setwd("/Users/nathalieguibert/Desktop/ResAss_Klüser_FS25/Ambizione Nathalie/Sexually Explicit Deepfakes")

# maybe use python to flatten better 
# not entirely flattened -> need to be at level of comments
json_flattened <- fromJSON("Posts_carahuntermla.json", flatten = TRUE)

json_flattenedch <- fromJSON("Posts_carahuntermla.json",flatten = TRUE)


setwd("/Users/nathalieguibert/Desktop/ResAss_Klüser_FS25/Ambizione Nathalie/Sexually Explicit Deepfakes/Code")
test <- read_csv("test1_flattened_comments.csv")
# 968 obvservations comments (including replies)
# 80 posts
# 452 unique comments
# 33 unique replies


dup_post_ids <- as.data.frame(test$post_id[duplicated(test$post_id)])


dup_comm_ids <- as.data.frame(test$comment_id[duplicated(test$comment_id)])


dup_rep_ids <- as.data.frame(test$reply_id[duplicated(test$reply_id)])







df_dup <- read_csv("df_duplicates.csv")



# Identify post_ids with duplicated comments
dupe_post_ids <- df_dup %>%
  filter(duplicate_comment == TRUE) %>%
  pull(post_id) %>%
  unique()

# Mark each row TRUE if its post_id is in the list of duplicate posts
df_dup <- df_dup %>%
  mutate(dup_post = post_id %in% dupe_post_ids)
# View the result
print(posts_with_duplicate_comments)



