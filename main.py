from textblob import TextBlob
import sys
import tweepy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import nltk
import pycountry
import re
import string
import time

from wordcloud import WordCloud, STOPWORDS
from PIL import Image
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langdetect import detect
from nltk.stem import SnowballStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer

api = tweepy.Client(
    bearer_token="AAAAAAAAAAAAAAAAAAAAACXMlgEAAAAAr%2F4yPAeHrgZptrVssrAo5foJp00%3DODpZbD9d5ItzQpLCoNlLc9tMFtNFnydCfa50CNmaEy4Um2N3JV")


def percentage(num, tot):
    return 100 * float(num) / float(tot)


keyword = input("Enter keyword to analyze: ")
noOfTweet = int(input("Enter number of tweets to analyze: "))

positive = 0
negative = 0
neutral = 0
polarity = 0
tweet_list = []
neutral_list = []
negative_list = []
positive_list = []

# s_time = time.time()
for tweet in tweepy.Paginator(api.search_recent_tweets, query=keyword).flatten(limit=noOfTweet):

    # print(tweet.text)
    tweet_list.append(tweet.text)
    # analysis = TextBlob(tweet.text)
    score = SentimentIntensityAnalyzer().polarity_scores(tweet.text)
    neg = score['neg']
    neu = score['neu']
    pos = score['pos']
    comp = score['compound']
    # polarity += analysis.sentiment.polarity

    if neg > pos:
        negative_list.append(tweet.text)
        negative += 1

    elif pos > neg:
        positive_list.append(tweet.text)
        positive += 1

    elif pos == neg:
        neutral_list.append(tweet.text)
        neutral += 1
# print("Total time taken: ", time.time() - s_time)

positive = percentage(positive, noOfTweet)
negative = percentage(negative, noOfTweet)
neutral = percentage(neutral, noOfTweet)
polarity = percentage(polarity, noOfTweet)
positive = format(positive, '.2f')
negative = format(negative, '.2f')
neutral = format(neutral, '.2f')

tweet_list = pd.DataFrame(tweet_list)
neutral_list = pd.DataFrame(neutral_list)
negative_list = pd.DataFrame(negative_list)
positive_list = pd.DataFrame(positive_list)
# print("Total tweets: ",len(tweet_list))
# print("Positive tweets: ",len(positive_list))
# print("Negative tweets: ", len(negative_list))
# print("Neutral tweets: ",len(neutral_list))

labels = ['Positive [' + str(positive) + '%]', 'Neutral [' + str(neutral) + '%]', 'Negative [' + str(negative) + '%]']
sizes = [positive, neutral, negative]
colors = ['yellowgreen', 'blue', 'red']
patches, texts = plt.pie(sizes, colors=colors, startangle=90)
plt.style.use('default')
plt.legend(labels)
plt.title("Sentiment Analysis for " + keyword + "")
plt.axis('equal')
plt.show()

tweet_list.drop_duplicates(inplace=True)
tw_list = pd.DataFrame(tweet_list)
tw_list["text"] = tw_list[0]

# print(tw_list)
# print(tweet_list)

tw_list = pd.DataFrame(tweet_list)
tw_list["text"] = tw_list[0]

# Removing RT, Punctuation etc
remove_rt = lambda x: re.sub('RT @\w+: ', " ", x)
rt = lambda x: re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", x)
tw_list["text"] = tw_list.text.map(remove_rt).map(rt)
tw_list["text"] = tw_list.text.str.lower()
# print(tw_list.head(10))

tw_list[['polarity', 'subjectivity']] = tw_list['text'].apply(lambda Text: pd.Series(TextBlob(Text).sentiment))
for index, row in tw_list['text'].items():
    score = SentimentIntensityAnalyzer().polarity_scores(row)
    neg = score['neg']
    neu = score['neu']
    pos = score['pos']
    comp = score['compound']
    if neg > pos:
        tw_list.loc[index, 'sentiment'] = "negative"
    elif pos > neg:
        tw_list.loc[index, 'sentiment'] = "positive"
    else:
        tw_list.loc[index, 'sentiment'] = "neutral"
    tw_list.loc[index, 'neg'] = neg
    tw_list.loc[index, 'neu'] = neu
    tw_list.loc[index, 'pos'] = pos
    tw_list.loc[index, 'compound'] = comp

# print(tw_list.head(10))

tw_list_negative = tw_list[tw_list["sentiment"] == "negative"]
tw_list_positive = tw_list[tw_list["sentiment"] == "positive"]
tw_list_neutral = tw_list[tw_list["sentiment"] == "neutral"]


def count_values_in_column(data, feature):
    total = data.loc[:, feature].value_counts(dropna=False)
    percentage = round(data.loc[:, feature].value_counts(dropna=False, normalize=True) * 100, 2)
    return pd.concat([total, percentage], axis=1, keys=['Total', 'Percentage'])


count_values_in_column(tw_list, "sentiment")

tw_list['text_len'] = tw_list['text'].astype(str).apply(len)
tw_list['text_word_count'] = tw_list['text'].apply(lambda x: len(str(x).split()))

round(pd.DataFrame(tw_list.groupby("sentiment").text_len.mean()), 2)

round(pd.DataFrame(tw_list.groupby("sentiment").text_word_count.mean()), 2)


def remove_punct(text):
    text = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text


tw_list['punct'] = tw_list['text'].apply(lambda x: remove_punct(x))


# Appliyng tokenization
def tokenization(text):
    text = re.split('\W+', text)
    return text


tw_list['tokenized'] = tw_list['punct'].apply(lambda x: tokenization(x.lower()))

stopword = nltk.corpus.stopwords.words('english')


def remove_stopwords(text):
    text = [word for word in text if word not in stopword]
    return text


tw_list['nonstop'] = tw_list['tokenized'].apply(lambda x: remove_stopwords(x))

ps = nltk.PorterStemmer()


def stemming(text):
    text = [ps.stem(word) for word in text]
    return text


tw_list['stemmed'] = tw_list['nonstop'].apply(lambda x: stemming(x))

character = []
for i in range(26):
    stopword.append(chr(97 + i))


def clean_text(text):
    text_lc = "".join([word.lower() for word in text if word not in string.punctuation])  # remove puntuation
    text_rc = re.sub('[0-9]+', '', text_lc)
    text_rmc = re.sub('\b\w{2}\b', '', text_rc)
    tokens = re.split('\W+', text_rmc)  # tokenization
    text = [ps.stem(word) for word in tokens if word not in stopword]  # remove stopwords and stemming
    return text


countVectorizer = CountVectorizer(analyzer=clean_text)
countVector = countVectorizer.fit_transform(tw_list['text'])
print('{} Number of reviews has {} words'.format(countVector.shape[0], countVector.shape[1]))
# print(countVectorizer.get_feature_names())

count_vect_df = pd.DataFrame(countVector.toarray(), columns=countVectorizer.get_feature_names_out())

count = pd.DataFrame(count_vect_df.sum())
countdf = count.sort_values(0, ascending=False).head(20)
# countdf[1:11]

countVectorizer = CountVectorizer(analyzer=clean_text)
countVectorn = countVectorizer.fit_transform(tw_list_negative['text'])
print('{} Number of reviews has {} words'.format(countVectorn.shape[0], countVectorn.shape[1]))

count_vect_n_df = pd.DataFrame(countVectorn.toarray(), columns=countVectorizer.get_feature_names_out())
# count_vect_n_df.head()

count = pd.DataFrame(count_vect_n_df.sum())
countndf = count.sort_values(0, ascending=False).head(20)
# countndf[1:11]

countVectorizer = CountVectorizer(analyzer=clean_text)
countVectorp = countVectorizer.fit_transform(tw_list_positive['text'])
print('{} Number of reviews has {} words'.format(countVectorp.shape[0], countVectorp.shape[1]))

count_vect_p_df = pd.DataFrame(countVectorp.toarray(), columns=countVectorizer.get_feature_names_out())

count = pd.DataFrame(count_vect_p_df.sum())
countpdf = count.sort_values(0, ascending=False).head(20)
