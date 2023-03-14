import streamlit as st
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
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langdetect import detect
from nltk.stem import SnowballStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer

st.set_page_config(layout="wide", page_title="Twitter Sentiment Analysis", page_icon="🐦")

st.title("Twitter Sentiment Analysis")

api = tweepy.Client(
    bearer_token="AAAAAAAAAAAAAAAAAAAAACXMlgEAAAAA0absjWEoeqhKkAhzwCgnzFbU860%3Dah35JDHAhLrQG5gth1tZGq8Y1XvmLFrUPRtOLwzvIssy6vHpIy", wait_on_rate_limit=True)


def percentage(num, tot):
    return 100 * float(num) / float(tot)

with st.form("input_form", clear_on_submit=True):
    keyword = st.text_input("Enter keyword to analyze: ")
    noOfTweet = st.number_input("Enter number of tweets to analyze: ", min_value=1, max_value=10000)
    textb = st.checkbox("Use TextBlob (Default = Vader)", value=False, key="blob")
    submit1 = st.form_submit_button("Submit")

positive = 0
negative = 0
neutral = 0
polarity = 0
tweet_list = []
neutral_list = []
negative_list = []
positive_list = []

if submit1 and keyword and not textb:
    print("Using Vader Sentiment Analysis")
    for tweet in tweepy.Paginator(api.search_recent_tweets, query=keyword).flatten(limit=noOfTweet):

        # print(tweet.text)
        tweet_list.append(tweet.text)
    #     analysis = TextBlob(tweet.text)
    #     score = SentimentIntensityAnalyzer().polarity_scores(tweet.text)
    #     neg = score['neg']
    #     neu = score['neu']
    #     pos = score['pos']
    #     comp = score['compound']
    #     polarity += analysis.sentiment.polarity
    #
    #     if neg > pos:
    #         negative_list.append(tweet.text)
    #         negative += 1
    #
    #     elif pos > neg:
    #         positive_list.append(tweet.text)
    #         positive += 1
    #
    #     elif pos == neg:
    #         neutral_list.append(tweet.text)
    #         neutral += 1
    #
    # positive = percentage(positive, noOfTweet)
    # negative = percentage(negative, noOfTweet)
    # neutral = percentage(neutral, noOfTweet)
    # polarity = percentage(polarity, noOfTweet)
    # positive = format(positive, '.2f')
    # negative = format(negative, '.2f')
    # neutral = format(neutral, '.2f')

    # print("done")

    tweet_list = pd.DataFrame(tweet_list)
    neutral_list = pd.DataFrame(neutral_list)
    negative_list = pd.DataFrame(negative_list)
    positive_list = pd.DataFrame(positive_list)
    # print("Total tweets: ",len(tweet_list))
    # print("Positive tweets: ",len(positive_list))
    # print("Negative tweets: ", len(negative_list))
    # print("Neutral tweets: ",len(neutral_list))


    # def show_raw_data():
    #     st.subheader("Raw Data")
    #     st.write(tweet_list)
    #
    # show_raw_data()
    col1, col2, col3 = st.columns(3)

    # with col1:
    #     labels = ['Positive [' + str(positive) + '%]', 'Neutral [' + str(neutral) + '%]',
    #               'Negative [' + str(negative) + '%]']
    #     sizes = [positive, neutral, negative]
    #     colors = ['green', 'gray', 'red']
    #     patches, texts = plt.pie(sizes, colors=colors, startangle=90)
    #     plt.style.use('default')
    #     plt.legend(labels)
    #     plt.title("Sentiment Analysis for \"" + keyword.upper() + "\"")
    #     plt.axis('equal')
    #     st.pyplot(plt)

    tweet_list.drop_duplicates(inplace=True)

    tw_list = pd.DataFrame(tweet_list)
    tw_list["text"] = tw_list[0]

    # Creating new dataframe and new features
    tw_list = pd.DataFrame(tweet_list)
    tw_list["text"] = tw_list[0]

    # Removing RT, Punctuation etc
    remove_rt = lambda x: re.sub('RT @\w+: ', " ", x)
    rt = lambda x: re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", x)
    tw_list["text"] = tw_list.text.map(remove_rt).map(rt)
    tw_list["text"] = tw_list.text.str.lower()

    tw_list[['polarity', 'subjectivity']] = tw_list['text'].apply(lambda Text: pd.Series(TextBlob(Text).sentiment))

    tot1, neg1, pos1, neu1 = 0, 0, 0, 0
    for index, row in tw_list['text'].items():
        tot1 += 1
        score = SentimentIntensityAnalyzer().polarity_scores(row)
        neg = score['neg']
        neu = score['neu']
        pos = score['pos']
        comp = score['compound']
        if neg > pos:
            tw_list.loc[index, 'sentiment'] = "negative"
            neg1 += 1
        elif pos > neg:
            tw_list.loc[index, 'sentiment'] = "positive"
            pos1 += 1
        else:
            tw_list.loc[index, 'sentiment'] = "neutral"
            neu1 += 1
        tw_list.loc[index, 'neg'] = neg
        tw_list.loc[index, 'neu'] = neu
        tw_list.loc[index, 'pos'] = pos
        tw_list.loc[index, 'compound'] = comp

    tw_list_negative = tw_list[tw_list["sentiment"] == "negative"]
    tw_list_positive = tw_list[tw_list["sentiment"] == "positive"]
    tw_list_neutral = tw_list[tw_list["sentiment"] == "neutral"]


    def count_values_in_column(data, feature):
        total = data.loc[:, feature].value_counts(dropna=False)
        percentage = round(data.loc[:, feature].value_counts(dropna=False, normalize=True) * 100, 2)
        return pd.concat([total, percentage], axis=1, keys=['Total', 'Percentage'])


    count_values_in_column(tw_list, "sentiment")


    posper = (pos1 / tot1) * 100
    negper = (neg1 / tot1) * 100
    neuper = (neu1 / tot1) * 100
    posper = format(posper, '.2f')
    negper = format(negper, '.2f')
    neuper = format(neuper, '.2f')

    with col2:
        labels = ['Positive [' + str(posper) + '%]', 'Neutral [' + str(neuper) + '%]',
                  'Negative [' + str(negper) + '%]']
        size = [pos1, neu1, neg1]
        names = 'Positive', 'Neutral', 'Negative'
        my_circle = plt.Circle((0, 0), 0.7, color='white')
        plt.clf()
        plt.pie(size, labels=names, colors=['green', 'blue', 'red'])
        p = plt.gcf()
        p.gca().add_artist(my_circle)
        plt.legend(labels)
        # plt.show()
        st.pyplot(plt)


    col4, col5, col6 = st.columns(3)


    def create_wordcloud(text):
        mask = np.array(Image.open("cloud.png"))
        stopwords = set(STOPWORDS)
        wc = WordCloud(background_color="white",
                       mask=mask,
                       max_words=3000,
                       stopwords=stopwords,
                       repeat=True)
        wc1 = wc.generate(str(text))
        # wc.to_file("wc.png")
        # print("Word Cloud Saved Successfully")
        # path = "wc.png"
        # display(Image.open(path))
        # st.image(wc1)
        plt.clf()
        plt.imshow(wc1, interpolation='bilinear')

        plt.title("")
        plt.legend()
        plt.axis("off")
        plt.show()
        st.pyplot(plt)
    # with col4:
    #     st.write("Word Cloud for all tweets")
    #     create_wordcloud(tw_list["text"].values)
    #
    # with col5:
    #     st.write("Word Cloud for negative tweets")
    #     create_wordcloud(tw_list_negative["text"].values)
    #
    # with col6:
    #     st.write("Word Cloud for positive tweets")
    #     create_wordcloud(tw_list_positive["text"].values)

    tw_list['text_len'] = tw_list['text'].astype(str).apply(len)
    tw_list['text_word_count'] = tw_list['text'].apply(lambda x: len(str(x).split()))

    round(pd.DataFrame(tw_list.groupby("sentiment").text_len.mean()), 2)

    round(pd.DataFrame(tw_list.groupby("sentiment").text_word_count.mean()), 2)


    # Removing Punctuation
    def remove_punct(text):
        text = "".join([char for char in text if char not in string.punctuation])
        text = re.sub('[0-9]+', '', text)
        return text


    tw_list['punct'] = tw_list['text'].apply(lambda x: remove_punct(x))


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

    wn = nltk.WordNetLemmatizer()

    def lemmatization(text):
        text = [wn.lemmatize(word) for word in text]
        return text

    tw_list['lemmatized'] = tw_list['nonstop'].apply(lambda x: lemmatization(x))

    def clean_text(text):
        text_lc = "".join([word.lower() for word in text if word not in string.punctuation])  # remove puntuation
        text_rc = re.sub('[0-9]+', '', text_lc)
        tokens = re.split('\W+', text_rc)  # tokenization
        text = [ps.stem(word) for word in tokens if word not in stopword]  # remove stopwords and stemming
        return text


    countVectorizer = CountVectorizer(analyzer=clean_text)
    countVector = countVectorizer.fit_transform(tw_list['text'])
    print('{} reviews has {} words'.format(countVector.shape[0], countVector.shape[1]))

    count_vect_df = pd.DataFrame(countVector.toarray(), columns=countVectorizer.get_feature_names_out())

    count = pd.DataFrame(count_vect_df.sum())
    countdf = count.sort_values(0, ascending=False).head(20)
    # uncomment to show the top 10 words
    # countdf[1:11]

    countVectorizer = CountVectorizer(analyzer=clean_text)
    countVectorn = countVectorizer.fit_transform(tw_list_negative['text'])
    print('{} reviews has {} words'.format(countVectorn.shape[0], countVectorn.shape[1]))

    count_vect_n_df = pd.DataFrame(countVectorn.toarray(), columns=countVectorizer.get_feature_names_out())

    count = pd.DataFrame(count_vect_n_df.sum())
    countndf = count.sort_values(0, ascending=False).head(20)

    countVectorizer = CountVectorizer(analyzer=clean_text)
    countVectorp = countVectorizer.fit_transform(tw_list_positive['text'])
    print('{} reviews has {} words'.format(countVectorp.shape[0], countVectorp.shape[1]))

    count_vect_p_df = pd.DataFrame(countVectorp.toarray(), columns=countVectorizer.get_feature_names_out())

    count = pd.DataFrame(count_vect_p_df.sum())
    countpdf = count.sort_values(0, ascending=False).head(20)


    def get_top_n_gram(corpus, ngram_range, n=None):
        vec = CountVectorizer(ngram_range=ngram_range, stop_words='english').fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        return words_freq[:n]


    col7, col8, col9 = st.columns(3)

    with col7:
        plt.clf()
        st.write("Top 10 bigrams for all tweets")
        common_words = get_top_n_gram(tw_list['text'], (3, 3), 10)
        for word, freq in common_words:
            st.write(word, freq)
            # print(word, freq)
        df1 = pd.DataFrame(common_words, columns=['text', 'count'])
        df1.groupby('text').sum()['count'].sort_values(ascending=False).plot(
            kind='bar', title='Top 10 bigrams for all tweets')
        # plt.show()
        st.pyplot(plt)

    with col8:
        plt.clf()
        st.write("Top 10 bigrams for negative tweets")
        common_words = get_top_n_gram(tw_list_negative['text'], (3, 3), 10)
        for word, freq in common_words:
            st.write(word, freq)
            # print(word, freq)
        df1 = pd.DataFrame(common_words, columns=['text', 'count'])
        df1.groupby('text').sum()['count'].sort_values(ascending=False).plot(
            kind='bar', title='Top 10 bigrams for negative tweets')
        # plt.show()
        st.pyplot(plt)

    with col9:
        plt.clf()
        st.write("Top 10 bigrams for positive tweets")
        common_words = get_top_n_gram(tw_list_positive['text'], (3, 3), 10)
        for word, freq in common_words:
            st.write(word, freq)
            # print(word, freq)
        df1 = pd.DataFrame(common_words, columns=['text', 'count'])
        df1.groupby('text').sum()['count'].sort_values(ascending=False).plot(
            kind='bar', title='Top 10 bigrams for positive tweets')
        # plt.show()
        st.pyplot(plt)


if textb and keyword and submit1:
    print("Using TextBlob")
    for tweet in tweepy.Paginator(api.search_recent_tweets, query=keyword).flatten(limit=noOfTweet):

        # print(tweet.text)
        tweet_list.append(tweet.text)
    #     analysis = TextBlob(tweet.text)
    #     score = SentimentIntensityAnalyzer().polarity_scores(tweet.text)
    #     neg = score['neg']
    #     neu = score['neu']
    #     pos = score['pos']
    #     comp = score['compound']
    #     polarity += analysis.sentiment.polarity
    #
    #     if neg > pos:
    #         negative_list.append(tweet.text)
    #         negative += 1
    #
    #     elif pos > neg:
    #         positive_list.append(tweet.text)
    #         positive += 1
    #
    #     elif pos == neg:
    #         neutral_list.append(tweet.text)
    #         neutral += 1
    #
    # positive = percentage(positive, noOfTweet)
    # negative = percentage(negative, noOfTweet)
    # neutral = percentage(neutral, noOfTweet)
    # polarity = percentage(polarity, noOfTweet)
    # positive = format(positive, '.2f')
    # negative = format(negative, '.2f')
    # neutral = format(neutral, '.2f')

    print("Done")
    tweet_list = pd.DataFrame(tweet_list)
    neutral_list = pd.DataFrame(neutral_list)
    negative_list = pd.DataFrame(negative_list)
    positive_list = pd.DataFrame(positive_list)
    # print("Total tweets: ",len(tweet_list))
    # print("Positive tweets: ",len(positive_list))
    # print("Negative tweets: ", len(negative_list))
    # print("Neutral tweets: ",len(neutral_list))


    # def show_raw_data():
    #     st.subheader("Raw Data")
    #     st.write(tweet_list)
    #
    # show_raw_data()
    col1, col2, col3 = st.columns(3)

    # with col1:
    #     labels = ['Positive [' + str(positive) + '%]', 'Neutral [' + str(neutral) + '%]',
    #               'Negative [' + str(negative) + '%]']
    #     sizes = [positive, neutral, negative]
    #     colors = ['green', 'gray', 'red']
    #     patches, texts = plt.pie(sizes, colors=colors, startangle=90)
    #     plt.style.use('default')
    #     plt.legend(labels)
    #     plt.title("Sentiment Analysis for \"" + keyword.upper() + "\"")
    #     plt.axis('equal')
    #     st.pyplot(plt)

    # print("Remove duplicates")
    tweet_list.drop_duplicates(inplace=True)
    # print("Done")

    tw_list = pd.DataFrame(tweet_list)
    tw_list["text"] = tw_list[0]

    # Creating new dataframe and new features
    tw_list = pd.DataFrame(tweet_list)
    tw_list["text"] = tw_list[0]

    # Removing RT, Punctuation etc
    # print("Remove pumctuation")
    remove_rt = lambda x: re.sub('RT @\w+: ', " ", x)
    rt = lambda x: re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", x)
    tw_list["text"] = tw_list.text.map(remove_rt).map(rt)
    tw_list["text"] = tw_list.text.str.lower()
    # print(done)


    tot1, neg1, pos1, neu1 = 0, 0, 0, 0
    # print("here")
    for index, row in tw_list['text'].items():
        tot1 += 1
        analysis = TextBlob(row)
        tw_list.loc[index, 'polarity'] = analysis.sentiment.polarity
        if analysis.sentiment.polarity > 0:
            tw_list.loc[index, 'sentiment'] = "positive"
            pos1 += 1
        elif analysis.sentiment.polarity == 0:
            tw_list.loc[index, 'sentiment'] = "neutral"
            neu1 += 1
        else:
            tw_list.loc[index, 'sentiment'] = "negative"
            neg1 += 1

    # print("done")

        # score = SentimentIntensityAnalyzer().polarity_scores(row)
        # neg = score['neg']
        # neu = score['neu']
        # pos = score['pos']
        # comp = score['compound']
        # if neg > pos:
        #     tw_list.loc[index, 'sentiment'] = "negative"
        #     neg1 += 1
        # elif pos > neg:
        #     tw_list.loc[index, 'sentiment'] = "positive"
        #     pos1 += 1
        # else:
        #     tw_list.loc[index, 'sentiment'] = "neutral"
        #     neu1 += 1
        # tw_list.loc[index, 'neg'] = neg
        # tw_list.loc[index, 'neu'] = neu
        # tw_list.loc[index, 'pos'] = pos
        # tw_list.loc[index, 'compound'] = comp

    tw_list_negative = tw_list[tw_list["sentiment"] == "negative"]
    tw_list_positive = tw_list[tw_list["sentiment"] == "positive"]
    tw_list_neutral = tw_list[tw_list["sentiment"] == "neutral"]


    def count_values_in_column(data, feature):
        total = data.loc[:, feature].value_counts(dropna=False)
        percentage = round(data.loc[:, feature].value_counts(dropna=False, normalize=True) * 100, 2)
        return pd.concat([total, percentage], axis=1, keys=['Total', 'Percentage'])


    count_values_in_column(tw_list, "sentiment")


    posper = (pos1 / tot1) * 100
    negper = (neg1 / tot1) * 100
    neuper = (neu1 / tot1) * 100
    posper = format(posper, '.2f')
    negper = format(negper, '.2f')
    neuper = format(neuper, '.2f')

    with col2:
        labels = ['Positive [' + str(posper) + '%]', 'Neutral [' + str(neuper) + '%]',
                  'Negative [' + str(negper) + '%]']
        size = [pos1, neu1, neg1]
        names = 'Positive', 'Neutral', 'Negative'
        my_circle = plt.Circle((0, 0), 0.7, color='white')
        plt.clf()
        plt.pie(size, labels=names, colors=['green', 'blue', 'red'])
        p = plt.gcf()
        p.gca().add_artist(my_circle)
        plt.legend(labels)
        # plt.show()
        st.pyplot(plt)


    col4, col5, col6 = st.columns(3)


    def create_wordcloud(text):
        mask = np.array(Image.open("cloud.png"))
        stopwords = set(STOPWORDS)
        wc = WordCloud(background_color="white",
                       mask=mask,
                       max_words=3000,
                       stopwords=stopwords,
                       repeat=True)
        wc1 = wc.generate(str(text))
        # wc.to_file("wc.png")
        # print("Word Cloud Saved Successfully")
        # path = "wc.png"
        # display(Image.open(path))
        # st.image(wc1)
        plt.imshow(wc1, interpolation='bilinear')
        plt.title("")
        plt.legend()
        plt.axis("off")
        plt.show()
        st.pyplot(plt)
    # with col4:
    #     st.write("Word Cloud for all tweets")
    #     create_wordcloud(tw_list["text"].values)
    #
    # with col5:
    #     st.write("Word Cloud for negative tweets")
    #     create_wordcloud(tw_list_negative["text"].values)
    #
    # with col6:
    #     st.write("Word Cloud for positive tweets")
    #     create_wordcloud(tw_list_positive["text"].values)

    tw_list['text_len'] = tw_list['text'].astype(str).apply(len)
    tw_list['text_word_count'] = tw_list['text'].apply(lambda x: len(str(x).split()))

    round(pd.DataFrame(tw_list.groupby("sentiment").text_len.mean()), 2)

    round(pd.DataFrame(tw_list.groupby("sentiment").text_word_count.mean()), 2)


    # Removing Punctuation
    def remove_punct(text):
        text = "".join([char for char in text if char not in string.punctuation])
        text = re.sub('[0-9]+', '', text)
        return text


    tw_list['punct'] = tw_list['text'].apply(lambda x: remove_punct(x))


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

    wn = nltk.WordNetLemmatizer()

    def lemmatization(text):
        text = [wn.lemmatize(word) for word in text]
        return text

    tw_list['lemmatized'] = tw_list['nonstop'].apply(lambda x: lemmatization(x))

    def clean_text(text):
        text_lc = "".join([word.lower() for word in text if word not in string.punctuation])  # remove puntuation
        text_rc = re.sub('[0-9]+', '', text_lc)
        tokens = re.split('\W+', text_rc)  # tokenization
        text = [ps.stem(word) for word in tokens if word not in stopword]  # remove stopwords and stemming
        return text


    countVectorizer = CountVectorizer(analyzer=clean_text)
    countVector = countVectorizer.fit_transform(tw_list['text'])
    print('{} reviews has {} words'.format(countVector.shape[0], countVector.shape[1]))

    count_vect_df = pd.DataFrame(countVector.toarray(), columns=countVectorizer.get_feature_names_out())

    count = pd.DataFrame(count_vect_df.sum())
    countdf = count.sort_values(0, ascending=False).head(20)
    # uncomment to show the top 10 words
    # countdf[1:11]

    countVectorizer = CountVectorizer(analyzer=clean_text)
    countVectorn = countVectorizer.fit_transform(tw_list_negative['text'])
    print('{} reviews has {} words'.format(countVectorn.shape[0], countVectorn.shape[1]))

    count_vect_n_df = pd.DataFrame(countVectorn.toarray(), columns=countVectorizer.get_feature_names_out())

    count = pd.DataFrame(count_vect_n_df.sum())
    countndf = count.sort_values(0, ascending=False).head(20)

    countVectorizer = CountVectorizer(analyzer=clean_text)
    countVectorp = countVectorizer.fit_transform(tw_list_positive['text'])
    print('{} reviews has {} words'.format(countVectorp.shape[0], countVectorp.shape[1]))

    count_vect_p_df = pd.DataFrame(countVectorp.toarray(), columns=countVectorizer.get_feature_names_out())

    count = pd.DataFrame(count_vect_p_df.sum())
    countpdf = count.sort_values(0, ascending=False).head(20)


    def get_top_n_gram(corpus, ngram_range, n=None):
        vec = CountVectorizer(ngram_range=ngram_range, stop_words='english').fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        return words_freq[:n]


    col7, col8, col9 = st.columns(3)

    with col7:
        plt.clf()
        st.write("Top 10 bigrams for all tweets")
        common_words = get_top_n_gram(tw_list['text'], (3, 3), 10)
        for word, freq in common_words:
            st.write(word, freq)
            # print(word, freq)
        df1 = pd.DataFrame(common_words, columns=['text', 'count'])
        df1.groupby('text').sum()['count'].sort_values(ascending=False).plot(
            kind='bar', title='Top 10 bigrams for all tweets')
        # plt.show()
        st.pyplot(plt)

    with col8:
        plt.clf()
        st.write("Top 10 bigrams for negative tweets")
        common_words = get_top_n_gram(tw_list_negative['text'], (3, 3), 10)
        for word, freq in common_words:
            st.write(word, freq)
            # print(word, freq)
        df1 = pd.DataFrame(common_words, columns=['text', 'count'])
        df1.groupby('text').sum()['count'].sort_values(ascending=False).plot(
            kind='bar', title='Top 10 bigrams for negative tweets')
        # plt.show()
        st.pyplot(plt)

    with col9:
        plt.clf()
        st.write("Top 10 bigrams for positive tweets")
        common_words = get_top_n_gram(tw_list_positive['text'], (3, 3), 10)
        for word, freq in common_words:
            st.write(word, freq)
            # print(word, freq)
        df1 = pd.DataFrame(common_words, columns=['text', 'count'])
        df1.groupby('text').sum()['count'].sort_values(ascending=False).plot(
            kind='bar', title='Top 10 bigrams for positive tweets')
        # plt.show()
        st.pyplot(plt)