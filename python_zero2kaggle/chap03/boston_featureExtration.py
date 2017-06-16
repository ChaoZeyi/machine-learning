# -*- coding:utf-8 -*-
from sklearn.datasets import fetch_20newsgroups
news = fetch_20newsgroups(subset='all')
from sklearn.cross_validation import train_test_split
type(news)
type(news.data)
x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25, random_state=33)
type(x_train)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x_train_count = cv.fit_transform(x_train)
type(x_train_count)
x_train_count
print x_train_count
