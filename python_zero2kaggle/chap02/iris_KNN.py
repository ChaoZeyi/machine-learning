# -*- coding:utf-8 -*-
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

iris = load_iris()
type(iris)
type(iris.data)
type(iris.target)
iris.data.shape
print iris.DESCR
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=33)

ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

kn = KNeighborsClassifier()
kn.fit(x_train, y_train)
kn.score(x_test, y_test)
y_predict = kn.predict(x_test)
print classification_report(y_test, y_predict, target_names=iris.target_names)
