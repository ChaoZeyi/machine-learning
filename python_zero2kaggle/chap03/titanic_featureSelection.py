# -*- coding:utf-8 -*-
import pandas as pd
titanic = pd.read_csv('F:\github\kaggle\Titanic\dataSets\\test.csv')
type(titanic)
titanic
x = titanic.drop(['row.names', 'name', 'survived'], axis = 0)
