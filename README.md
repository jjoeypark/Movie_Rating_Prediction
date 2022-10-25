# Movie_Rating_Prediction
Predict the star rating associated with user reviews from Amazon Movie Reviews using the available features.

## NUMERIC DATA
### 1.	Feature: "Helpfulness."

I assigned -1 to review with low helpfulness(O<val<0.35), O to medium helpfulness(3.5<= val< 0.75) and 1 to high helpfulness(0.75<=val<1). It is useful to understand whether the review was helpful or not.


### 2.	Feature: "Predict_score_mean" - "mean_processing.ipynb" file

If two or more identical users of the same product gave similar reviews, and in case to predict one of their scores, we could figure it out by comparing scores each other. To compare the ratings of each movie from each user, I thought SVD would help comparing similarities between users and scores. Using SVD, make user-item rating matrix into "user matrix X singular value matrix X movie matrix ". The reduced matrix can predict the rating for movies that the user did not evaluate. I tried to do latent factor collaborative filtering.
In a function called predict_by_svd(), I created a pivot table with "User Id" as an index, "Productld" as a column, and score as value, subtracted the entire pivot table by the average score of each user, and then divided the matrix into U, sigma, and vt via SVD. Then, converted sigma into a symmetric matrix through np.diag(). After making a new matrix with changed sigma, would finally get the expected score of each product. 
The predict_by_comparing_users() function is a function that stores userid and Productld in the entire test set to the list and creates an expected score list using for loop-using for loop to specify target user and product. Afterward, gathering the information about the other users who rated the target product from the train set, and make a matrix with the index of "users who watched the target movie and target user" and columns of "target users' product list".


### 3. Feature: "Product mean & user mean"

Rather than putting it in the average, I thought it would prevent overfitting by dividing it into integers, so I assigned integers to each mean: 1 to 5


### 4. Review Length

Through graph, I found the relations between the length of review and the score. If the score is 5 or 1 (which means they liked or hated that movie) they didn't need to talk much about it. However, when they were in the middle state(2, 3, 4), they had to explain why they deducted the stars. Because of this, I tried to put review length and I used MinMax Scalar to scale the result

### 5. Feature: DateTime
I made date to YYYYMM feature and then scaled it. 

## Text Data 
### 6. Feature: Text & Summary sentiment
Since there were missing values in the text and plot, I filled the NULL value in Summary, using "NONE" and used the text in Summary to fill the Text column. 
First, I created a review_to_words() function to tokenize the word using nltk and then extracted all the stems of the word through snowball stemmer. After that, I applied the review_to_words() function to all Text and Summary through for loop and got a clean text. 
After that, I used CountVectorizer to vectorize the words. As "Summary" has a small number of words comparing to the "Text", I set max_features to 6000, max_df = 0.5, and min_df = 2. I did this because if the stem is more than 0.5, it might be unnecessary. In addition, I set ngram_range, in which continuous words can be examined, as (1, 3). By using this, I could somewhat sort contextual negative and positive sentiment of the text. For example, **"it was not fun"** and **"it was fun"** could be sorted into different sentiments. Therefore, vectorized clean_train through vectorizer.fit_transform. In addition, Text designated max_features as 150000 and max_df as 0.6 
