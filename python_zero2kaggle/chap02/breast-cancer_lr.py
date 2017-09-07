# -*- coding : utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDClassifier, LogisticRegression
####线性回归，随机梯度下降分类，逻辑回归
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
column_names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion','Single Epithelial Cell Size' ,
    'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data", names=column_names)
type(data)
data.describe()
data.info()
data = data.replace(to_replace='?', value=np.nan)
data = data.dropna(how = 'any', axis=0)
data.shape
x_train, x_test, y_train, y_test = train_test_split(data[column_names[1:10]], data[column_names[10]], test_size=0.25, random_state=33)
y_train.value_counts()
y_test.value_counts()
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_train.mean(axis=0)
x_train.var(axis=0)
x_test = ss.transform(x_test)
lr = LinearRegression()
lr.fit(x_train, y_train)
y_predict1 = lr.predict(x_test)
sgd = SGDClassifier()
sgd.fit(x_train, y_train)
y_predict2 = sgd.predict(x_test)

print(lr.score(x_test, y_test))
print(classification_report(y_test, y_predict2))
sgd.score(x_test, y_test)
y_predict1.shape
y_test.shape
y_predict1
y_predict1[y_predict1 <= 3] = 2
y_predict1[y_predict1 > 3] = 4
y_predict1.shape
y_predict2
print(classification_report(y_test, y_predict1, target_names=['Benign', 'Malignant']))
ls = RandomForestClassifier()
ls.fit(x_train, y_train)
ls.feature_importances_119
