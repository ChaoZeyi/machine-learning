# -*- coding:utf-8 -*-
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
boston = load_boston()
type(boston)
print boston.DESCR

type(boston.data)
type(boston.target)

x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.25, random_state=33)

ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

y_train= ss.fit_transform(y_train)
y_test = ss.transform(y_test)

lr = LinearRegression()
lr.fit(x_train, y_train)
lr.score(x_test, y_test)
y_predict = lr.predict(x_test)

print mean_squared_error(ss.inverse_transform(y_test), ss.inverse_transform(y_predict))
print mean_absolute_error(ss.inverse_transform(y_test), ss.inverse_transform(y_predict))
