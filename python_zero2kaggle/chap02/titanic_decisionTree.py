# -*- coding:utf-8 -*-
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report

titanic = pd.read_csv("F:/github/kaggle/Titanic/dataSets/train.csv")


type(titanic)
titanic.shape
titanic.describe()
titanic.info()
titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())

titanic.loc[titanic['Sex']=='male', 'Sex'] = 0
titanic.loc[titanic['Sex']=='female', 'Sex'] = 1

x_titanic = titanic [['Pclass', 'Age', 'Sex', 'SibSp', 'Parch', 'Fare']]
y_titanic  = titanic ['Survived']

x_train, x_test, y_train, y_test = train_test_split(x_titanic, y_titanic, test_size=0.25, random_state=33)

ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_train.shape
x_test.shape
x_test = ss.transform(x_test)

dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
print(dt.score(x_test, y_test))
y_predict = dt.predict(x_test)
print(classification_report(y_test, y_predict))
print(dt.feature_importances_)
