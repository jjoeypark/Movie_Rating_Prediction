from numpy import NaN
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def process(df):
    # Fill empty datas 
    df['Text'].fillna(df.Summary, inplace = True)
    df['Summary'].fillna('NONE', inplace = True)

    # This is where you can do all your processing
    df['Helpfulness'] = df['HelpfulnessNumerator'] / df['HelpfulnessDenominator']
    df['Helpfulness'] = df['Helpfulness'].fillna(0)
    def helpfulness_preprocessing(value):
        if value >= 0 and value < 0.35:
            return -1
        elif value >= 0.35 and value <0.75:
            return 0
        else:
            return 1
    df['Helpfulness'] = df['Helpfulness'].apply(helpfulness_preprocessing)

    df['Predict_score_mean'] = pd.read_csv("predict_test.csv") 
    df['Predict_score_mean'].fillna(df['Score'], inplace = True)

    # Scaled Review Length feature 
    scalar = MinMaxScaler()
    df['ReviewLength'] = df.apply(lambda row : len(row['Text'].split()) if type(row['Text']) == str else 0, axis = 1)
    arr1 = df['ReviewLength'].values.reshape(-1, 1)
    df['ReviewLengthScale'] = pd.Series(scalar.fit_transform(arr1).reshape(-1))

    # Mean score of users and products 
    mean_score = df.Score.mean()
    df['user_mean_score'] = df.groupby('UserId').Score.transform('mean')
    df['user_mean_score'].fillna(mean_score, inplace = True)
    df['product_mean_score'] = df.groupby('ProductId').Score.transform('mean')
    df['product_mean_score'].fillna(mean_score, inplace = True)

    def score_preprocessing(value):
        if value >= 1.0 and value < 1.8:
            return 1
        elif value >= 1.8 and value <2.6:
            return 2
        elif value >= 2.6 and value <3.4:
            return 3
        elif value >= 3.4 and value <4.2:
            return 4
        else:
            return 5

    print("Processing...")    
    
    df['product_mean_score'] = df['product_mean_score'].apply(score_preprocessing)
    df['user_mean_score'] = df['user_mean_score'].apply(score_preprocessing)


    # Feature for YYYY(Year)MM(Month) 
    trainDate = pd.to_datetime(df['Time'], unit='s')
    df['YYYYMM'] = trainDate.dt.year*100 + trainDate.dt.month
    arr2 = df['YYYYMM'].values.reshape(-1, 1)
    df['DateScale'] = pd.Series(scalar.fit_transform(arr2).reshape(-1))

    # # Feature that indicates if the user is reliable by using the number of reviews and helpfulness
    # df['reliableUser'] = (df.user_num_reviews > 5) & (df.Helpfulness >= 0.6)

    # Feature that indicates the score with the max HelpfulnessNumerator (which got the highest helpful score)
    df_helpful = df[df['Helpfulness'] > 0.5]
    helpfulnumer = df_helpful.groupby(['ProductId', 'Score']).HelpfulnessNumerator.sum().reset_index()
    findMaxHelp = pd.DataFrame(helpfulnumer.groupby('ProductId', as_index= False).HelpfulnessNumerator.idxmax())
    maxHelpScore = []
    for index, row in findMaxHelp.iterrows():
        m = helpfulnumer.iloc[int(row['HelpfulnessNumerator'])]['Score']
        maxHelpScore.append(int(m))
    findMaxHelp['maxHelpNum'] = maxHelpScore
    addCol = findMaxHelp.drop('HelpfulnessNumerator', axis = 1)
    df = df.merge(addCol, on='ProductId', how = 'left')
    df = df.rename(columns={'maxHelpNum_y': 'maxHelpNum'})    
    df['maxHelpNum'].fillna(5, inplace = True)

    s = df[['Score', 'Id']]
    recentData = pd.DataFrame(df.groupby('ProductId')['YYYYMM'].nlargest(100))
    recentData= recentData.reset_index(drop=False)
    recentData = recentData.rename(columns = {'level_1': 'Id'})
    recentData = recentData.merge(s, on= 'Id')
    recentMean = pd.DataFrame(recentData.groupby('ProductId')['Score'].mean())
    recentMean = recentMean.rename(columns={'Score': 'recent_prod_mean'})
    df = df.merge(recentMean, on= 'ProductId')
    df['recent_prod_mean'].fillna(mean_score, inplace = True)

   # Feature that indicates the sentiment of the review 
    def score_sentiments(value):
        if value <= 2:
            return int(-1)
        elif value == 3:
            return int(0)
        elif value > 3:
            return int(1)
        else: 
            return NaN

    df['textSentiment'] = df['Score'].apply(score_sentiments)
    df['summSentiment'] = df['Score'].apply(score_sentiments)

    # For nulls in this feature I will put the result from the prediction using nltk tokenize, countvectorizer, and random forest
    textSentiment_test = pd.read_csv("textSentiment1.csv")
    df['textSentiment'] = df['textSentiment'].fillna(df['Id'].map(textSentiment_test.set_index('Id')['Sentiment']))

    summSentiment_test = pd.read_csv("summarySentiment1.csv") 
    df['summSentiment'] = df['summSentiment'].fillna(df['Id'].map(summSentiment_test.set_index('Id')['Sentiment']))

    #after all preprocessing I thought the difference for sentiment was too small so I changed the difference by apply function
    def sentiment_change(value):
        if value == -1:
            return -5
        elif value == 0:
            return 0
        else:
            return 5

    df['textSentiment'] = df['textSentiment'].apply(sentiment_change)
    df['summSentiment'] = df['summSentiment'].apply(sentiment_change)

    print("Almost Done...")    
    df = df.astype({'maxHelpNum': int, 'textSentiment': int, 'summSentiment':int})
    df1 = df.drop(['Time', 'HelpfulnessNumerator', 'HelpfulnessDenominator', 'YYYYMM'], axis = 1)
    return df1


# Load the dataset
trainingSet = pd.read_csv("./data/train.csv")

# Process the DataFrame
train_processed = process(trainingSet)

# Load test set
submissionSet = pd.read_csv("./data/test.csv")

# Merge on Id so that the test set can have feature columns as well
testX= pd.merge(train_processed, submissionSet, left_on='Id', right_on='Id')
testX = testX.drop(columns=['Score_x'])
testX = testX.rename(columns={'Score_y': 'Score'})

# The training set is where the score is not null
trainX =  train_processed[train_processed['Score'].notnull()]

testX.to_csv("./data/X_test.csv", index=False)
trainX.to_csv("./data/X_train.csv", index=False)

