# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 11:28:36 2021

@author: Siddhesh.Dosi
"""

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

def get_top_frequency_n_gram(corpus,ngram_range,n=None,stop_words=None):
    vec = CountVectorizer(ngram_range=ngram_range,stop_words = stop_words).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

def get_top_tfidf_n_gram(corpus,ngram_range,n=None,stop_words=None):
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=ngram_range)
    vz = vectorizer.fit_transform(corpus)
    words_freq = [(word,score) for word,score in zip(vectorizer.get_feature_names(), vectorizer.idf_)]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
