# -*- coding:utf-8 -*-
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
x_train = [[6], [8], [10], [14], [18]]
type(x_train)
y_train = [[7], [9], [13], [17.5], [18]]
xx =np.linspace(0,26,100)
type(xx)
xx
xx = xx.reshape(xx.shape[0], 1)
type(xx)
xx.shape
pf = PolynomialFeatures()
x_train_pf = pf.fit_transform(x_train)

lr = LinearRegression()
lr.fit(x_train, y_train)

lr_pf = LinearRegression()
lr_pf.fit(x_train_pf, y_train)
xx_pf = pf.transform(xx)
yy_pf = lr_pf.predict(xx_pf)
xx.shape
yy = lr.predict(xx)
yy.shape

plt.scatter(x_train, y_train)
plt1,= plt.plot(xx, yy, label="Degree1")
plt2,= plt.plot(xx, yy_pf, label="Degree2")
plt.axis([0,25,0,25])
plt.xlabel('Diameter of Pizza')
plt.ylabel('price of pizza')
plt.legend()
plt.show()
print(lr.score(x_train, y_train))
print(lr_pf.score(x_train_pf, y_train))
