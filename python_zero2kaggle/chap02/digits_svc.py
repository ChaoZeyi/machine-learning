# -*- coding:utf-8 -*-
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

digits = load_digits()
type(digits)
digits.DESCR
type(digits.data)
type(digits.target)
digits.data.shape
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=33)
type(x_test)
x_test.shape
y_train.shape

#############对训练和测试数据进行标准化######################################
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

lsvc = LinearSVC()
lsvc.fit(x_train, y_train)
lsvc.score(x_test, y_test)
y_predict = lsvc.predict(x_test)
print classification_report(y_test, y_predict)


svc = SVC()
svc.fit(x_train, y_train)
svc.score(x_test, y_test)
