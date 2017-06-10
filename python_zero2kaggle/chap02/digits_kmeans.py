# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import silhouette_score
digits_train = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra', header=None)
digits_test = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes', header=None)

type(digits_train)
digits_train
x_train = digits_train[np.arange(64)]
y_train = digits_train[64]

x_test = digits_test[np.arange(64)]
y_test = digits_test[64]

km = KMeans(n_clusters=10)
dir(km)
km.n_clusters
km.fit(x_train)
y_predict = km.predict(x_test)
y_predict
print adjusted_rand_score(y_test, y_predict)
print silhouette_score(x_test, y_predict)
