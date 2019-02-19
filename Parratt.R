#############################################################
# title: "Movie Prediction Capstone Project"
# author: "Steffen Parratt"
# date: "2/18/2019"
#############################################################
# WARNING: Please read through this file before running it.
# Towards the end of the file there are some models that run
# for hours and may crash your computer due to memory constraints.
# These pieces of code have a warning label before they appear.
#############################################################
# OUTLINE OF FILE:
# * Data Gathering
# * Data Preparation
# * Model Selection
# * Parameter Optimization: Regularization
# * Model Training & Evaluation
# * Other Factors onsidered: Exploration & Visualization
# * Other Modeling Approaches
# * Results
#############################################################
# DATA GATHERING
#
# Beginning of the setup code provided by edX 
# PLEASE NOTE: THIS CODE HAS NOT BEEN EDITED IN ANY WAY
# Create edx set, validation set, and submission file
# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                      col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

#############################################################
# DATA PREPARATION
# 
# Purpose is to ensure the dataset is complete and error-free
# Temporarily combine edx and validation data sets for checking...
data_set <- rbind(edx, validation)

# Check to see that columns are the types we expect
column_classes_are_incorrect <- !is.integer(data_set$userId) | 
                                !is.numeric(data_set$movieId) |
                                !is.numeric(data_set$rating) |
                                !is.integer(data_set$timestamp) |
                                !is.character(data_set$title) |
                                !is.character(data_set$genres)
if (column_classes_are_incorrect) print("Unexpected column classes")

# Check to see that there are no NA values in the data fields
NAs_in_row <- data_set %>% filter(is.na(userId) | 
                                  is.na(movieId) | 
                                  is.na(rating) | 
                                  is.na(timestamp) | 
                                  is.na(title)   |
                                  is.na(genres))
NAs_present_in_data_set <- (nrow(NAs_in_row) > 0)  
if (NAs_present_in_data_set) print("Investigate NA values in data")

# Check to see that movieId fields are integer values
movieId_not_integer_row <- data_set %>% filter(movieId%%1 !=0)
movieId_not_integer <- (nrow(movieId_not_integer_row) > 0) 
if (movieId_not_integer) print("Investigate non-integer movieId fields")

# Check to see that ratings are between 0 and 5
rating_data_error <- data_set %>% filter(rating < 0.0 | rating > 5.0)
ratings_out_of_range <- (nrow(rating_data_error) > 0) 
if (ratings_out_of_range) print("Data cleansing required to correct userId rows")

# Determine whether to stop the analysis because of corrupt data
if (column_classes_are_incorrect |
    NAs_present_in_data_set |
    movieId_not_integer |
    ratings_out_of_range) { 
  print("DATA PREPARATION SUMMARY: Stop analysis and cleanse data.") 
  } else {
  print("DATA PREPARATION SUMMARY: Data clean, continue with analysis.")
}

# If no data issues, then remove temporary data set
rm(data_set)

# ... and continue to the next phase of analysis...

#############################################################
# MODEL and PARAMETER SELECTION
#
# Start with the system outlined in the course textbook....

# Partition the data into training and test sets
set.seed(1)
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
train_set <- edx[-test_index,]
test_set <- edx[test_index,]

# To ensure we do not include users and movies in the test set that do not appear in 
# the training set we remove those entries with the semi_join function
test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId") 
  
# We will test the accuracy of our model with the RMSE function
RMSE <- function(predicted_ratings, true_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Similar to the textbook, our first model predicts the same rating for all movies 
# regardless of the user
mu <- mean(train_set$rating)
mu # 3.512482

# If we predict all unknown ratings with mu we obtain the following RMSE:
naive_rmse <- RMSE(mu, test_set$rating)
naive_rmse # 1.059904

# Let's create a results table and add our first entry
rmse_results <- tibble(method = "Just the average", RMSE = naive_rmse)

# We augment our model by adding term b_i to represent the average ranking for movie i.
# b_i is the average least square estimate of the difference between Y - mu for each movie.
movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

# Let’s see how much our prediction improves once we add b_i to our model
predicted_ratings <- mu + test_set %>% 
  left_join(movie_avgs, by='movieId') %>% 
  pull(b_i)

model_1_rmse <- RMSE(predicted_ratings, test_set$rating)
model_1_rmse # 0.9437429

rmse_results <- bind_rows(rmse_results, tibble(method="Movie Effect Model", RMSE = model_1_rmse))

# Now we augment our previous model by adding the term b_u to represent the average ranking for user u.
user_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# We can now construct predictors and see how much the RMSE improves:
predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

model_2_rmse <- RMSE(predicted_ratings, test_set$rating) # 0.866
rmse_results <- bind_rows(rmse_results, tibble(method="Movie + User Effects Model", RMSE = model_2_rmse))
rmse_results
# A tibble: 3 x 2
# method                      RMSE
#  <chr>                      <dbl>
#1 Just the average           1.06 
#2 Movie Effect Model         0.944
#3 Movie + User Effects Model 0.866

# I ran this for different seeds, and there was no noticeable impact on results
# This RMSE is below the threshold of 0.87750, which gives full points for this exercise

#############################################################
# PARAMETER OPTIMIZATION: REGULARIZATION
#
# Now we try to improve our model by adding regularization
# We need to choose a parameter lambda that optimizes the regularization of effects

lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l){
  
  mu <- mean(train_set$rating)
  
  b_i <- train_set %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, test_set$rating))
})

plot(lambdas, rmses)
lambda <- lambdas[which.min(rmses)]
lambda # 4.75
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Regularized Movie + User Effect Model",  
                                     RMSE = min(rmses)))
rmse_results %>% knitr::kable()
# |method                                |      RMSE|
#  |:-------------------------------------|---------:|
#  |Just the average                      | 1.0599043|
#  |Movie Effect Model                    | 0.9437429|
#  |Movie + User Effects Model            | 0.8659320|
#  |Regularized Movie + User Effect Model | 0.8652421|
#
# From this table we can that regularization adds little value

# Below we will review some other model enhancements that I chose not to include,
# but first we will run our model on the validation set and record our final RMSE.

#############################################################
# MODEL TRAINING & EVALUATION
#
# Using the model and analysis above, we repeat the analyis for the full edx and 
# validation data sets that were provided
# Below is my FINAL MODEL for grading

mu <- mean(edx$rating)
lambda <- 4.75

b_i <- edx %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda))

b_u <- edx %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))

predicted_ratings <- validation %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

final_rmse <- RMSE(predicted_ratings, validation$rating)

rmse_results <- bind_rows(rmse_results,
                          tibble(method="Validation dataset",  
                                 RMSE = final_rmse))
final_results <- rmse_results
rmse_results %>% knitr::kable()

#  |:-------------------------------------|---------:|
#  |Just the average                      | 1.0599043|
#  |Movie Effect Model                    | 0.9437429|
#  |Movie + User Effects Model            | 0.8659320|
#  |Regularized Movie + User Effect Model | 0.8652421|
#  |Validation dataset                    | 0.8648201|

# These results are described in the "results" section of the report

#############################################################
# OTHER FACTORS TO CONSIDER: GENRES
#
# In studying the Movielens dataset, we saw evidence of a "genre effect",
# which we can see in the plot
test_set %>%
  group_by(genres) %>%
  summarize(n = n(), avg = mean(rating), se = sd(rating)/sqrt(n())) %>%
  filter(n >= 1000) %>% 
  mutate(genres = reorder(genres, avg)) %>%
  ggplot(aes(x = genres, y = avg, ymin = avg - 2*se, ymax = avg + 2*se)) + 
  geom_point() +
  geom_errorbar() + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# Let's go back and reset our training and test sets
set.seed(1)
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
train_set <- edx[-test_index,]
test_set <- edx[test_index,]

# To ensure we do not include genres in the test set that do not appear in 
# the training set we remove those entries with the semi_join function
test_set <- test_set %>% 
  semi_join(train_set, by = "genres") 

# Again, mu is set as the mean rating
mu <- mean(train_set$rating)
mu # 3.512482

naive_rmse <- RMSE(mu, test_set$rating)
naive_rmse # 1.059909

# We compute term b_g to represent the average ranking for genres g.
genres_avgs <- train_set %>% 
  group_by(genres) %>% 
  summarize(b_g = mean(rating - mu))

# Let’s see how much our prediction improves once we add b_g to our model
predicted_ratings <- mu + test_set %>% 
  left_join(genres_avgs, by='genres') %>% 
  mutate(pred = mu + b_g) %>%
  pull(b_g)

model_genres_rmse <- RMSE(predicted_ratings, test_set$rating)
model_genres_rmse # 1.01781
genres_rmse_impact <- naive_rmse - model_genres_rmse
genres_rmse_impact # 0.042099

# We can see that the impact of the genres effect is relatively minor
# Since our validation set has not been defined to ensure the presence
# of all genres, we do not include this effect in our model

#############################################################
# OTHER FACTORS TO CONSIDER: DATE
#
# In studying the Movielens dataset, we saw some evidence of a "time effect",
# which we can see in the plot
library(lubridate)
train_set %>% mutate(date = round_date(as_datetime(timestamp), unit = "week")) %>%
  group_by(date) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(date, rating)) +
  geom_point() +
  geom_smooth()

# First, we need to add a "date" column to the data set
edx_with_dates <- edx %>% mutate(date = round_date(as_datetime(timestamp), unit = "week"))

# Then we need to repartition our dataset into training and test sets
set.seed(1)
test_index <- createDataPartition(y = edx_with_dates$rating, times = 1, p = 0.2, list = FALSE)
train_set <- edx_with_dates[-test_index,]
test_set <- edx_with_dates[test_index,]

# To ensure we do not include dates in the test set that do not appear in 
# the training set we remove those entries with the semi_join function
test_set <- test_set %>% 
  semi_join(train_set, by = "date") 

mu <- mean(train_set$rating) 
mu # 3.512482

naive_rmse <- RMSE(mu, test_set$rating)
naive_rmse # 1.059909

# We compute term b_d to represent the average ranking for date d.
date_avgs <- train_set %>% 
  group_by(date) %>% 
  summarize(b_d = mean(rating - mu))

predicted_ratings <- mu + test_set %>% 
  left_join(date_avgs, by='date') %>% 
  pull(b_d)

model_date_rmse <- RMSE(predicted_ratings, test_set$rating)
model_date_rmse # 1.056202

date_rmse_impact <- naive_rmse - model_date_rmse
date_rmse_impact # 0.003707791

# The impact of the date effect is almost negligible.
# Since our validation set has not been defined to ensure the presence
# of all timestamps, we do not include this effect in our model

#############################################################
# OTHER MODELING APPROACHES
#
# WARNING!!! --- the code below may process for hours and crash
# your computer. Please proceed at your own risk!
#
# "Recommenderlab" was mentioned in our course textbook. 
# It provides matrix factorization tools.

install.packages("recommenderlab")
library(recommenderlab)

# First, but the data set in sparse matrix format
sparse_ratings <- sparseMatrix(i = edx$userId, j = edx$movieId, x = edx$rating,
                               dims = c(max(edx$userId), max(edx$movieId)),  
                               dimnames = list(paste("u", 1:max(edx$userId), sep = ""), 
                                               paste("m", 1:max(edx$movieId), sep = "")))
  
# Second, eliminate any rows that are blank (otherwise recommenderlab will fail)                             
sparse_ratings <- sparse_ratings[which(rowSums(sparse_ratings) > 0),] 

# Convert the sparse matrix into recommenderlab's realRatingMatrix format
real_ratings <- new("realRatingMatrix", data = sparse_ratings)

# Set the seed and compute the evaluation scheme
# WARNING -- this can take hours, and may fail because of memory issues
set.seed(1)
e <- evaluationScheme(real_ratings, method="split", train=0.8, given=-5)
# 5 ratings of 20% of users are excluded for testing

# Let's first try the Popular model
model <- Recommender(getData(e, "train"), "POPULAR")

# And make our prediction from the "known" training data
prediction <- predict(model, getData(e, "known"), type="ratings")

# Then compute the RMSE of that model
rmse_popular <- calcPredictionAccuracy(prediction, getData(e, "unknown"))
rmse_popular # 0.9234480

model <- Recommender(getData(e, "train"), method = "UBCF",
                     parameter=list(normalize = "center", method="Cosine", nn=50))
prediction <- predict(model, getData(e, "known"), type="ratings")
# Error in asMethod(object) : 
#   Cholmod error 'problem too large' at file ../Core/cholmod_dense.c, line 105
# "The error statement appears to confirm that you have a memory issue. 
# You'll most-likely need more memory to analyze that dataset as-is."
rmse_ubcf <- calcPredictionAccuracy(prediction, getData(e, "unknown"))
rmse_ubcf 

model <- Recommender(getData(e, "train"), method = "IBCF", 
                     parameter=list(normalize = "center", method="Cosine", k=350))
prediction <- predict(model, getData(e, "known"), type="ratings")
# Error in asMethod(object) : 
#   Cholmod error 'problem too large' at file ../Core/cholmod_dense.c, line 105
# "The error statement appears to confirm that you have a memory issue. 
# You'll most-likely need more memory to analyze that dataset as-is."
rmse_ibcf <- calcPredictionAccuracy(prediction, getData(e, "unknown"))
rmse_ibcf

#############################################################
# RESULTS
#
final_results %>% knitr::kable()
#
#  |method                                |      RMSE|
#  |:-------------------------------------|---------:|
#  |Just the average                      | 1.0599043|
#  |Movie Effect Model                    | 0.9437429|
#  |Movie + User Effects Model            | 0.8659320|
#  |Regularized Movie + User Effect Model | 0.8652421|
#  |Validation dataset                    | 0.8648170|


#############################################################