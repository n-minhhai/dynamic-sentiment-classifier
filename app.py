
import twint
import pandas as pd
import numpy as np
import re
import pickle
import streamlit as st
import matplotlib.pyplot as plt
import asyncio
import os


## Extract tweets using TWINT
def extract_tweets(user_input, earliest_year, latest_year, limit):
    print('... Extracting Tweets ...')
    earliest_year = earliest_year + '-01-01 00:00:00'
    c = twint.Config()
    c.Search = user_input
    c.Since = earliest_year
    c.Year = latest_year
    c.Limit = limit
    c.Hide_output = True
    c.Output = '{}.csv'.format(user_input)
    twint.run.Search(c)
    print('... Tweets Extracted Successfully ...')
    pass



## Load the tweets as a dataframe and check for any required preprocessing
def load_data(user_input):
    df = pd.read_csv('{}.csv'.format(user_input), names=['tweet'], delimiter='\n')
    ## Use regex to remove tweet ID, date, time, user, and mentions - extract ONLY  the tweet content
    df['tweet'] = df['tweet'].str.replace(r"(\d+)|([--])|([::])|([+])|(<\S+)|(@\S+)|", "")
    ## Remove whitespaces in the beginning
    df['tweet'] = df['tweet'].str.replace(r"^\s+", "")
    
    print('... Tweets Loaded Successfully ...')
    return df


## Perform sentiment prediction using vectorizer and trained classifier
def sentiment_predict(df):
    x = df['tweet']

    batches = np.array_split(np.array(x),20)
    positive = 0
    negative = 0

    vectorizer = pickle.load(open('model/vectorizer.pickle', 'rb'))
    model = pickle.load(open('model/classifier.pickle', 'rb'))

    ## Make sentiment prediction
    for batch in batches:
        batch = vectorizer.transform(batch).toarray()
        prediction = model.predict(batch)
        positive = positive + sum(prediction==2)
        negative = negative + sum(prediction==0)

    positive_percentage = 100*positive / x.shape[0]
    negative_percentage = 100*negative / x.shape[0]
    print('Percent of positive sentiment: {:.2f}%'.format(positive_percentage))
    print('Percent of negative sentiment: {:.2f}%'.format(negative_percentage))
    print('... Sentiment prediction finishing ...')
    return positive_percentage, negative_percentage, int(x.shape[0])

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)


st.header("Tweet Sentiment Classification")

user_input = st.text_input("Please enter tweet to analyse:")

if user_input:
    earliest_year = st.text_input("Please enter the earliest year of the tweets: ")
    if earliest_year:
        latest_year = st.text_input("Please enter the latest year of the tweets (not inclusive): ")
        if latest_year:
            n = st.text_input("Limit number of tweets to extract (exceeding 8k may take a long time): ")
            if n:
                st.write('Loading sentiment classification for {} . . .'.format(user_input))
                extract_tweets(user_input, earliest_year, latest_year, n)
                df = load_data(user_input)
                positive, negative, n_tweets = sentiment_predict(df)
                st.write('Percentage of positive: {:.2f} \nPercentage of negative: {:.2f}'.format(positive, negative))
                st.write('Number of tweets used: ', n_tweets)

                plt.bar(['Positive', 'Negative'], [positive, negative], align='center')
                plt.xlabel('Sentiment')
                plt.ylabel('Percentage')
                plt.title('{} {}'.format(user_input, earliest_year))
                st.pyplot()

                os.remove('{}.csv'.format(user_input))
                print('Done\n')


