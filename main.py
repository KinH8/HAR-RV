# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 16:35:39 2022

@author: Wu
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from arch import arch_model

def HAR(df):
    df = pd.DataFrame(df)
    df[1] = df[0].rolling(5, min_periods=5).mean()
    df[2] = df[0].rolling(21, min_periods=21).mean()
    
    df.dropna(inplace=True)
    X = df.values[:-1]
    y = df.shift(-1)[0].values[:-1]
    prediction_var = df.values[-1]
    
    reg = LinearRegression().fit(X,y)

    return reg.predict(prediction_var.reshape(1,-1))
    
def garch(x):
    am = arch_model(x*1000, vol='Garch',p=1,o=0,q=1,dist='Normal')
    res = am.fit(update_freq=5, disp='off')
    forecasts = res.forecast(reindex=False).variance.iloc[0]/np.square(1000)
    return np.sqrt(forecasts)

rawdata = pd.read_csv('IK1.csv', header=None, index_col=[0], parse_dates=True)
rawdata.index.name = 'Date'
rawdata.columns = ['Realized variance']

# Calculate 5-min RV
data = rawdata.resample('5min').last()
data = data.groupby(data.index.date).apply(lambda x:x.pct_change().apply(np.square).sum())

data = data.mask(data==0, np.nan).dropna()
data['Realized vol'] = data['Realized variance']**0.5

# HAR model
data['HAR'] =  data['Realized vol'].rolling(21*2, min_periods=21*2).apply(HAR)

# Garch model
ret = rawdata['Realized variance'].resample('1d').last().pct_change().rename('Ret')
data = data.join(ret,how='left')

data['GARCH'] = np.nan
data['GARCH'] = data['Ret'].rolling(21, min_periods=21).apply(garch)
data['GARCH'] = data['GARCH'].shift()

# Compare to ewm of various time horizon
data['ewm5'] = data['Ret'].ewm(5).std().shift()
data['ewm32'] = data['Ret'].ewm(32).std().shift()

result = data[['Realized vol','HAR','GARCH','ewm5','ewm32']].dropna()
print(result.corr()**2)
result.plot()