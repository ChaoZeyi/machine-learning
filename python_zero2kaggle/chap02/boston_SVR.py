from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.svm import SVR, LinearSVR

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

svr = SVR(kernel="linear")
svr.fit(x_train, y_train)
svr.score(x_test, y_test)
y_predict = svr.predict(x_test)

print mean_squared_error(ss.inverse_transform(y_test), ss.inverse_transform(y_predict))
print mean_absolute_error(ss.inverse_transform(y_test), ss.inverse_transform(y_predict))

lsvr = LinearSVR()
lsvr.fit(x_train, y_train)
lsvr.score(x_test, y_test)
y_predict1 = lsvr.predict(x_test)

print mean_squared_error(ss.inverse_transform(y_test), ss.inverse_transform(y_predict1))
print mean_absolute_error(ss.inverse_transform(y_test), ss.inverse_transform(y_predict1))
