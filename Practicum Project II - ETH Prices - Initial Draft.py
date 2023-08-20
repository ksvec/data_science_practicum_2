#!/usr/bin/env python
# coding: utf-8

import yfinance
from datetime import datetime

get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')

api_key = API_KEY
symbol_data = ['ETH-USD','Ethereum']

#end date may -> june as validation data
df = yfinance.download(symbol_data[0],'2021-06-01', '2023-07-30')

df.head()
df.tail()
df.info()
df.describe()
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
plt.xlabel('Year')
plt.ylabel('Closure Price')
plt.plot(df['Close'])
plt.title(symbol_data[1]+' Price in the Last 2 Years')
plt.show()

# importing required packages
import pandas as pd  # Pandas for data manipulation
import pandas_profiling # Our Secret Sauce
from pandas_profiling import ProfileReport  # Generate Report

# Generate Entire EDA report with sigle line of code    
ProfileReport(df)

import numpy as np
dfclose = df['Close']

dflog = np.log(dfclose)

training_data, testing_data = dflog[3:int(len(dflog)*0.6)], dflog[int(len(dflog)*0.6):]

plt.figure(figsize=(10,8))
plt.xlabel('Dates')
plt.ylabel('Closing Prices')
plt.plot(dflog, 'green', label='Train data')
plt.plot(testing_data, 'blue', label='Test data')
plt.legend()

from statsmodels.tsa.arima.model import ARIMA

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import math
from sklearn.metrics import mean_squared_error

#grid_search sklearn help automates

# Assuming you have already fitted your model and stored it in `fitted_model`
# You should have a list or array of `testing_data` to make predictions on.

# Try different combinations of p, d, and q to find the best parameters
best_rmse = float('inf')
best_order = None

for p in range(4):  # You can adjust the range based on the maximum value you want to try for p
    for d in range(4):  # You can adjust the range based on the maximum value you want to try for d
        for q in range(4):  # You can adjust the range based on the maximum value you want to try for q
            model = ARIMA(training_data, order=(p, d, q))
            fitted_model = model.fit()

            # Make predictions on testing data
            fcast = fitted_model.forecast(steps=len(testing_data), alpha=0.05)

            # Calculate RMSE
            rmse = math.sqrt(mean_squared_error(testing_data, fcast))

            # Store the best parameters with the lowest RMSE
            if rmse < best_rmse:
                best_rmse = rmse
                best_order = (p, d, q)

model = ARIMA(training_data, order=best_order)

fitted_model = model.fit()

print(fitted_model.summary())

residuals = pd.DataFrame(fitted_model.resid)
fig, ax = plt.subplots(1,2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()

#fcast, se , conf = fitted_model.forecast(steps=len(testing_data), alpha=0.05)

fcast = fitted_model.forecast(steps=len(testing_data), alpha=0.05)

import pandas as pd
import matplotlib.pyplot as plt

# Adjust the code based on the structure of fcast
# For example, if the forecasted values and confidence intervals are returned as separate arrays
# you might need to modify the indexing as shown below.

if isinstance(fcast, tuple):
    fc_series = fcast[0]  # Assuming forecasted values are in the first element
    conf = fcast[2]       # Assuming confidence intervals are in the third element
else:
    # If `fcast` is not returned as a tuple, it might be a single array representing the forecasted values
    fc_series = fcast
    conf = None

# Create separate Series for the upper and lower confidence intervals
if conf is not None:
    lower_series = pd.Series(conf[:, 0], index=testing_data.index)
    upper_series = pd.Series(conf[:, 1], index=testing_data.index)

# Convert the forecasted values to a Pandas Series
fc_series = pd.Series(fc_series, index=testing_data.index)

# Plotting the forecast and confidence intervals if available
plt.figure(figsize=(10, 8))
if conf is not None:
    plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=0.09)
plt.plot(testing_data.index, fc_series, label='Forecast', color='blue')
plt.plot(testing_data.index, testing_data, label='Actual', color='green')
plt.xlabel('Dates')
plt.ylabel('Closing Prices')
plt.title('Prediction of '+symbol_data[1]+ ' Price')
plt.legend()
plt.show()

fcast
testing_data.mean()
training_data
testing_data

#LTSM model good results . memory
#C-suite level visualization and decisions/impact off that

