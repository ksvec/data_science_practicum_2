#!/usr/bin/env python
# coding: utf-8
# author: kyle

# Import necessary libraries
import yfinance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
import math
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense
from statsmodels.tsa.api import SimpleExpSmoothing

#Suppress/ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Download data using Yahoo Finance API
api_key = '9BTZQJA8HHVIH64EMVXJ2M4C9XH16KT5W5'
symbol_data = ['ETH-USD', 'Ethereum']
df = yfinance.download(symbol_data[0], '2021-06-01', '2023-07-30')

# Initial data exploration
df.info()
df.describe()

# Plot the closing prices
plt.figure(figsize=(10, 8))
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.plot(df['Close'])
plt.title(symbol_data[1] + ' Price in the Last 2 Years')
plt.show()

# Log transformation of the closing prices
dfclose = df['Close']
dflog = np.log(dfclose)

# Split data into training and testing sets
training_data, testing_data = dflog[3:int(len(dflog) * 0.6)], dflog[int(len(dflog) * 0.6):]

# Plot the training and testing data
plt.figure(figsize=(10, 8))
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.plot(dflog, 'green', label='Train data')
plt.plot(testing_data, 'blue', label='Test data')
plt.title('Train vs Test Data - ETH Prices')
plt.legend()

# Find the best ARIMA model parameters
best_rmse = float('inf')
best_order = None

for p in range(4):
    for d in range(4):
        for q in range(4):
            model = ARIMA(training_data, order=(p, d, q))
            fitted_model = model.fit()
            fcast = fitted_model.forecast(steps=len(testing_data), alpha=0.05)
            rmse = math.sqrt(mean_squared_error(testing_data, fcast))
            if rmse < best_rmse:
                best_rmse = rmse
                best_order = (p, d, q)

# Fit the best ARIMA model
model = ARIMA(training_data, order=best_order)
fitted_model = model.fit()
print(fitted_model.summary())

# Plot residuals and density
residuals = pd.DataFrame(fitted_model.resid)
fig, ax = plt.subplots(1, 2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()

# Make ARIMA predictions
fcast = fitted_model.forecast(steps=len(testing_data), alpha=0.05)

# Plot ARIMA predictions with confidence intervals
if isinstance(fcast, tuple):
    fc_series = fcast[0]
    conf = fcast[2]
else:
    fc_series = fcast
    conf = None

if conf is not None:
    lower_series = pd.Series(conf[:, 0], index=testing_data.index)
    upper_series = pd.Series(conf[:, 1], index=testing_data.index)

fc_series = pd.Series(fc_series, index=testing_data.index)

plt.figure(figsize=(10, 8))
if conf is not None:
    plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=0.09)
plt.plot(testing_data.index, fc_series, label='Forecast', color='blue')
plt.plot(testing_data.index, testing_data, label='Actual', color='green')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Prediction of ' + symbol_data[1] + ' Price - ARIMA Model')
plt.legend()
plt.show()

#Create RMSE and MAPE result archive
rmse_mape_results = []

# Calculate RMSE and MAPE for ARIMA Model 
rmse = mean_squared_error(np.exp(testing_data), np.exp(fcast), squared=False) # Back-transform to original scale
mape = mean_absolute_percentage_error(np.exp(testing_data), np.exp(fcast)) # Back-transform to original scale
print("RMSE: ", rmse)
print("MAPE: ", mape)

rmse_mape_results.append(['ARIMA', rmse, mape])

# LSTM modeling
def lstm_split(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps + 1):
        X.append(data[i:i + n_steps, :-1])
        y.append(data[i + n_steps - 1, -1])
    return np.array(X), np.array(y)

X_feat = df.iloc[:,0:4]

X1, y1 = lstm_split(X_feat.values, n_steps=2)
train_split = 0.8
split_idx = int(np.ceil(len(X1) * train_split))
date_index = X_feat.index

X_train, X_test = X1[:split_idx], X1[split_idx:]
y_train, y_test = y1[:split_idx], y1[split_idx:]
X_train_date, X_test_date = date_index[:split_idx], date_index[split_idx:]

# Create and compile the LSTM model
lstm = Sequential()
lstm.add(LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2]), activation='relu', return_sequences=True))
lstm.add(LSTM(32, activation='relu'))
lstm.add(Dense(1))
lstm.compile(loss='mean_squared_error', optimizer='adam')

# Train the LSTM model
history = lstm.fit(X_train, y_train, epochs=100, batch_size=4, verbose=2, shuffle=False)

# Make predictions using the LSTM model
y_pred = lstm.predict(X_test)

# Copy the X_test_date set for graphing y values
y_date_index = X_test_date  

# Plot LSTM predictions vs. actual values
plt.figure(figsize=(10, 8))
plt.plot(y_date_index[1:len(y_date_index)], y_pred, label='Forecast', color='blue')
plt.plot(y_date_index[1:len(y_date_index)], y_test, label='Actual', color='green')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Prediction of ' + symbol_data[1] + ' Price - LSTM Model #1')
plt.legend()
plt.show()

# Calculate RMSE and MAPE for LSTM Model #1
rmse = mean_squared_error(y_pred, y_test, squared=False)
mape = mean_absolute_percentage_error(y_pred, y_test)
print("RSME: ", rmse)
print("MAPE: ", mape)

rmse_mape_results.append(['LSTM #1', rmse, mape])

# Create and train another LSTM model with different parameters
lstm = Sequential()
lstm.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2]), activation='relu', return_sequences=True))
lstm.add(LSTM(50, activation='relu'))
lstm.add(Dense(1))
lstm.compile(loss='mean_squared_error', optimizer='adam')
history = lstm.fit(X_train, y_train, epochs=100, batch_size=4, verbose=2, shuffle=False)

# Make predictions using the second LSTM model
y_pred = lstm.predict(X_test)

# Plot LSTM predictions vs. actual values for the second model
plt.figure(figsize=(10, 8))
plt.plot(y_date_index[1:len(y_date_index)], y_pred, label='Forecast', color='blue')
plt.plot(y_date_index[1:len(y_date_index)], y_test, label='Actual', color='green')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Prediction of ' + symbol_data[1] + ' Price - LSTM Model #2')
plt.legend()
plt.show()

# Calculate RMSE and MAPE for LSTM Model #2
rmse = mean_squared_error(y_pred, y_test, squared=False)
mape = mean_absolute_percentage_error(y_pred, y_test)
print("RSME: ", rmse)
print("MAPE: ", mape)

rmse_mape_results.append(['LSTM #2', rmse, mape])

# Create and train a third LSTM model with different parameters
X1, y1 = lstm_split(X_feat.values, n_steps=10)
train_split = 0.8
split_idx = int(np.ceil(len(X1) * train_split))
date_index = X_feat.index
X_train, X_test = X1[:split_idx], X1[split_idx:]
y_train, y_test = y1[:split_idx], y1[split_idx:]
X_train_date, X_test_date = date_index[:split_idx], date_index[split_idx:]

lstm = Sequential()
lstm.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2]), activation='relu', return_sequences=True))
lstm.add(LSTM(50, activation='relu'))
lstm.add(Dense(1))
lstm.compile(loss='mean_squared_error', optimizer='adam')
history = lstm.fit(X_train, y_train, epochs=100, batch_size=4, verbose=2, shuffle=False)

# Make predictions using the third LSTM model
y_pred = lstm.predict(X_test)

# Plot LSTM predictions vs. actual values for the third model
plt.figure(figsize=(10, 8))
plt.plot(y_date_index[2:len(y_date_index)], y_pred, label='Forecast', color='blue')
plt.plot(y_date_index[2:len(y_date_index)], y_test, label='Actual', color='green')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Prediction of ' + symbol_data[1] + ' Price - LSTM Model #3')
plt.legend()
plt.show()

# Calculate RMSE and MAPE for LSTM Model #3
rmse = mean_squared_error(y_test, y_pred, squared=False)
mape = mean_absolute_percentage_error(y_test, y_pred)
print("RSME: ", rmse)
print("MAPE: ", mape)

rmse_mape_results.append(['LSTM #3', rmse, mape])

# Simple Moving Average
train_split = 0.8
split_idx = int(np.ceil(len(X_feat) * train_split))
train = X_feat[['Close']].iloc[:split_idx]
test = X_feat[['Close']].iloc[split_idx:]

# Calculate the Simple Moving Average
test_pred = np.array([train.rolling(10).mean().iloc[-1]] * len(test)).reshape((-1, 1))

# Plot Simple Moving Average vs. actual values
plt.figure(figsize=(10, 8))
plt.plot(test.index, test)
plt.plot(test.index, test_pred)
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Simple Moving Average - ETH')
plt.show()

print('RMSE: %.3f' % mean_squared_error(test, test_pred, squared=False))
print('MAPE: %.3f' % mean_absolute_percentage_error(test, test_pred))

rmse = mean_squared_error(test, test_pred, squared=False)
mape = mean_absolute_percentage_error(test, test_pred)

rmse_mape_results.append(['SMA', rmse, mape])

# Exponential Moving Average
X = X_feat[['Close']].values
train_split = 0.8
split_idx = int(np.ceil(len(X) * train_split))
train = X[:split_idx]
test = X[split_idx:]
test_concat = np.array([]).reshape((0, 1))

# Calculate Exponential Moving Average
for i in range(len(test)):
    train_fit = np.concatenate((train, np.asarray(test_concat)))
    fit = SimpleExpSmoothing(np.asarray(train_fit)).fit(smoothing_level=0.2)
    test_pred = fit.forecast(1)
    test_concat = np.concatenate((np.asarray(test_concat), test_pred.reshape((-1, 1))))

# Plot Exponential Moving Average vs. actual values
plt.figure(figsize=(10, 8))
plt.plot(test)
plt.plot(test_concat)
plt.xlabel('Epoch')
plt.ylabel('Close Price')
plt.title('Exponential Moving Average - ETH')
plt.show()

print('RMSE: %.3f' % mean_squared_error(test, test_concat, squared=False))
print('MAPE: %.3f' % mean_absolute_percentage_error(test, test_concat))

rmse = mean_squared_error(test, test_concat, squared=False)
mape = mean_absolute_percentage_error(test, test_concat)

rmse_mape_results.append(['EMA', rmse, mape])

#Create dataframe with various model RMSE and MAPE values
df = pd.DataFrame(rmse_mape_results, columns = ['Model', 'RMSE', 'MAPE'])
df.sort_values(by='RMSE', ascending=False, inplace=True)
print('\n')
print(df.to_string(index=False))
