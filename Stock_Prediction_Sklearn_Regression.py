#Import requirements
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib as mpl
import math
import numpy as np
from pandas import Series, DataFrame

#pip install pandas-datareader
import pandas_datareader as web
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

start = datetime.datetime(2010, 1, 1)
end = datetime.datetime.now()
#!pip install yfinance --upgrade --no-cache-dir

df = web.DataReader("AAPL", 'yahoo', start, end)

close_px = df['Adj Close']
mavg = close_px.rolling(window=100).mean()

mpl.rc('figure', figsize=(8, 7))
mpl.__version__

# Adjusting the style of matplotlib
style.use('ggplot')
close_px.plot(label='AAPL')
mavg.plot(label='mavg')
plt.legend()

rets = close_px / close_px.shift(1) - 1
rets.plot(label='return')

plt.show()

    
dfreg = df.loc[:, ['Adj Close', 'Volume']]
dfreg['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
dfreg['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0
print(dfreg.head())

# Drop missing value
dfreg.fillna(value=-99999, inplace=True)

print(dfreg.shape)

# We want to separate 2 percent of the data to forecast
forecast_out = int(math.ceil(0.02 * len(dfreg)))
print("FORECAST OUT")
print(forecast_out)  # 2% of 2442 = 49 Days

# Separating the label here, we want to predict the AdjClose
forecast_col = 'Adj Close'
dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
X = np.array(dfreg.drop(['label'], 1))

# Scale the X so that everyone can have the same distribution for linear regression
X = preprocessing.scale(X)

# Finally We want to find Data Series of late X and early X (train) for model generation and evaluation
X_lately = X[-forecast_out:]

X = X[:-forecast_out]

# Separate label and identify it as y
y = np.array(dfreg['label'])
y = y[:-forecast_out]

print('Dimension of X', X.shape)
print('Dimension of y', y.shape)

# Splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

train_len = len(X_train)
test_len = len(X_test)
print('train length', len(X_train))
print('test length', len(X_test))

# Linear regression
clfreg = LinearRegression(n_jobs=-2)
clfreg.fit(X_train, y_train)

# Quadratic Regression 2
clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
clfpoly2.fit(X_train, y_train)

# Quadratic Regression 3
clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
clfpoly3.fit(X_train, y_train)

# LassoCV
clfLasso = LassoCV(eps=0.002, n_alphas=100, fit_intercept=True, normalize=False)
clfLasso.fit(X_train, y_train)

# Score models
scores = []
modelNames = ["LinearRegression", "Quadratic2", "Quadratic3", "LassoCV"]
confidencereg = clfreg.score(X_test, y_test)
confidencepoly2 = clfpoly2.score(X_test, y_test)
confidencepoly3 = clfpoly3.score(X_test, y_test)
confLasso = clfLasso.score(X_test, y_test)
scores.append(confidencereg)
scores.append(confidencepoly2)
scores.append(confidencepoly3)
scores.append(confLasso)

# results
print("The linear regression confidence is: ", confidencereg)
print("The quadratic regression 2 confidence is: ", confidencepoly2)
print("The quadratic regression 3 confidence is: ", confidencepoly3)
print("The LassoCV confidence is: ", confLasso)
bsIndex = [i for i, j in enumerate(scores) if j == max(scores)]
print("BEST SCORE: " + str(max(scores)) + " WAS WITH MODEL:" + str(modelNames[bsIndex[0]]))

# Ploting
if (modelNames[bsIndex[0]] == "LinearRegression"):
    forecast = clfreg.predict(X_lately)
if (modelNames[bsIndex[0]] == "Quadratic2"):
    forecast = clfpoly2.predict(X_lately)
if (modelNames[bsIndex[0]] == "Quadratic3"):
    forecast = clfpoly3.predict(X_lately)
if (modelNames[bsIndex[0]] == "LassoCV"):
    forecast = clfLasso.predict(X_lately)

dfreg['ForecastReg'] = np.nan
print(forecast, confidencereg, forecast_out)

last_date = dfreg.iloc[-1].name
last_unix = last_date
next_unix = last_unix + datetime.timedelta(days=1)

for i in forecast:
    next_date = next_unix
    print(next_date)
    next_unix += datetime.timedelta(days=1)
    dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns) - 1)] + [i]

dfreg['Adj Close'].tail(1000).plot()
dfreg['ForecastReg'].tail(1000).plot()
plt.legend(loc=4)
plt.title("Price prediction with " + str(modelNames[bsIndex[0]]) + " Model")
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
