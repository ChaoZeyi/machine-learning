# -*- coding:utf-8 -*-
from sklearn.datasets import fetch_20newsgroups
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import numpy as np
#######运行该段代码，会自动从20newsgroups官网下载20news-bydate.pkz文件
#######官网：http://qwone.com/~jason/20Newsgroups/
#######默认下载保存路径：C:\Users\Administrator\scikit_learn_data
#######所以运行时会很长一段时间没有反应，是正常的
news = fetch_20newsgroups(subset='all')
type(news.target)
len(news.data)
news.target.shape
print news.data[0]
type(news.data)
x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25, random_state=33)
vec = CountVectorizer()
x_train = vec.fit_transform(x_train)
x_test = vec.transform(x_test)
mnb = MultinomialNB()
mnb.fit(x_train, y_train)
mnb.score(x_test, y_test)
y_predict = mnb.predict(x_test)
print classification_report(y_test, y_predict)
