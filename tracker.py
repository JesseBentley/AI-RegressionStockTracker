import pandas as pd
import quandl , math
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate

data = quandl.get('WIKI/GOOGL')
data = data[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
data['HL_PCT'] = (data['Adj. High'] - data['Adj. Close']) / data['Adj. Close'] * 100
data['PCT_change'] = (data['Adj. Close'] - data['Adj. Open']) / data['Adj. Open'] * 100
data = data[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
data.fillna(-9999, inplace = True)
forecast_out = int(math.ceil(0.01 * len(data)))

data['label'] = data[forecast_col].shift(-forecast_out)

X = np.array(data.drop('label'), 1)
X = preprocessing.scale(X)
X = X[: -forecast_out]
X_lately = X[-forecast_out:]
data.dropna(inplace = True)
y = np.array(data['label'])
y = np.array(data['label'])

X_train, X_test, y_train, y_test = cross_validate.train_test_split(X, y, test_size = 0.2)
clf = LinearRegression(n_jobs = -1)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

print(data.head)