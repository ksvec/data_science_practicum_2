#!/usr/bin/env python
# coding: utf-8

# In[1]:


import yfinance
from datetime import datetime
import pandas as pd

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


api_key = '9BTZQJA8HHVIH64EMVXJ2M4C9XH16KT5W5'

symbol_data = ['ETH-USD','Ethereum']

#end date may -> june as validation data

df = yfinance.download(symbol_data[0],'2021-06-01', '2023-07-30')
print(symbol_data[0])
df.head()
df.tail()


# In[3]:


df.info()


# In[4]:


df.describe()


# In[5]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
plt.xlabel('Year')
plt.ylabel('Close Price')
plt.plot(df['Close'])
plt.title(symbol_data[1]+' Price in the Last 2 Years')
plt.show()


# In[6]:


# importing required packages
#import pandas_profiling
#from pandas_profiling import ProfileReport  # Generate Report


# In[7]:


#Generate Entire EDA report with sigle line of code    
#ProfileReport(df)


# In[8]:


import numpy as np
dfclose = df['Close']

dflog = np.log(dfclose)


# In[9]:


training_data, testing_data = dflog[3:int(len(dflog)*0.6)], dflog[int(len(dflog)*0.6):]

plt.figure(figsize=(10,8))
plt.xlabel('Year')
plt.ylabel('Close Price')
plt.plot(dflog, 'green', label='Train data')
plt.plot(testing_data, 'blue', label='Test data')
plt.title('Train vs Test Data - ETH Prices')
plt.legend()


# In[10]:


import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.arima.model import ARIMA

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit

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


# In[11]:


import warnings
warnings.filterwarnings('ignore')

fitted_model = model.fit()

print(fitted_model.summary())


# In[12]:


residuals = pd.DataFrame(fitted_model.resid)
fig, ax = plt.subplots(1,2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()


# In[13]:


#fcast, se , conf = fitted_model.forecast(steps=len(testing_data), alpha=0.05)

fcast = fitted_model.forecast(steps=len(testing_data), alpha=0.05)


# In[14]:


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
plt.title('Prediction of '+symbol_data[1]+ ' Price - ARIMA Model')
plt.legend()
plt.show()


# In[15]:


fcast


# In[16]:


testing_data.mean()


# In[17]:


training_data


# In[18]:


testing_data


# In[19]:


#LTSM model good results . memory
#C-suite level visualization and decisions/impact off that


# In[20]:


#LTSM


# In[21]:


df = yfinance.download(symbol_data[0],'2021-06-01', '2023-07-30')
print(symbol_data[0])
df.head()
df.tail()


# In[22]:


target_y = df['Close']
X_feat = df.iloc[:,0:4]


# In[23]:


X_feat


# In[24]:


target_y


# In[25]:


#LSTM model
#increase Jupyter Notebook allocated resources

def lstm_split(data, n_steps):
    X, y = [],[]
    for i in range(len(data)-n_steps+1):
        X.append(data[i:i + n_steps, : -1])
        y.append(data[i + n_steps-1, -1])
        
    return np.array(X), np.array(y)


# In[26]:


X1, y1 = lstm_split(X_feat.values, n_steps=2)

train_split = 0.8
split_idx = int(np.ceil(len(X1)*train_split))
date_index = X_feat.index

X_train, X_test = X1[:split_idx], X1[split_idx:]
y_train, y_test = y1[:split_idx], y1[split_idx:]
X_train_date, X_test_date = date_index[:split_idx], date_index[split_idx:]

print(X1.shape, X_train.shape, X_test.shape, y_test.shape)


# In[27]:


import warnings
warnings.filterwarnings('ignore')

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.layers import Dense

lstm = Sequential()
lstm.add(LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2])
              ,activation='relu', return_sequences=True))
lstm.add(LSTM(32, activation='relu'))

lstm.add(Dense(1))
lstm.compile(loss='mean_squared_error', optimizer='adam')
lstm.summary()


# In[28]:


history=lstm.fit(X_train, y_train, epochs = 100, batch_size = 4,
                verbose = 2, shuffle = False)


# In[29]:


y_pred = lstm.predict(X_test)


# In[30]:


y_pred


# In[31]:


y_date_index = X_test_date  #copy the X_test_date set for graphing y values


# In[32]:


plt.plot(y_date_index[1:len(y_date_index)], 
         y_pred, label='Forecast', color='blue')
plt.plot(y_date_index[1:len(y_date_index)], 
         y_test, label='Actual', color='green')
plt.xlabel('Year')
plt.ylabel('Close Price')
plt.title('Prediction of '+symbol_data[1]+ ' Price - LSTM Model #1')
plt.legend()
plt.show()

rmse = mean_squared_error(y_pred,y_test, squared=False)
mape = mean_absolute_percentage_error(y_pred,y_test)
print("RSME: ", rmse)
print("MAPE: ", mape)


# In[37]:


lstm = Sequential()
lstm.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])
              ,activation='relu', return_sequences=True))
lstm.add(LSTM(50, activation='relu'))

lstm.add(Dense(1))
lstm.compile(loss='mean_squared_error', optimizer='adam')
lstm.summary()


# In[38]:


history=lstm.fit(X_train, y_train, epochs = 100, batch_size = 4,
                verbose = 2, shuffle = False)


# In[39]:


y_pred = lstm.predict(X_test)


# In[65]:


print(y_pred[0:5])
print(y_pred[(len(y_pred)-5):len(y_pred)])


# In[62]:


plt.plot(y_date_index[1:len(y_date_index)], 
         y_pred, label='Forecast', color='blue')
plt.plot(y_date_index[1:len(y_date_index)], 
         y_test, label='Actual', color='green')
plt.xlabel('Year')
plt.ylabel('Close Price')
plt.title('Prediction of '+symbol_data[1]+ ' Price - LSTM Model #2')
plt.legend()
plt.show()

rmse = mean_squared_error(y_pred,y_test, squared=False)
mape = mean_absolute_percentage_error(y_pred,y_test)
print("RSME: ", rmse)
print("MAPE: ", mape)


# In[46]:


X1, y1 = lstm_split(X_feat.values, n_steps=10)

train_split = 0.8
split_idx = int(np.ceil(len(X1)*train_split))
date_index = X_feat.index

X_train, X_test = X1[:split_idx], X1[split_idx:]
y_train, y_test = y1[:split_idx], y1[split_idx:]
X_train_date, X_test_date = date_index[:split_idx], date_index[split_idx:]

print(X1.shape, X_train.shape, X_test.shape, y_test.shape)


# In[47]:


lstm = Sequential()
lstm.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])
              ,activation='relu', return_sequences=True))
lstm.add(LSTM(50, activation='relu'))

lstm.add(Dense(1))
lstm.compile(loss='mean_squared_error', optimizer='adam')
lstm.summary()


# In[48]:


history=lstm.fit(X_train, y_train, epochs = 100, batch_size = 4,
                verbose = 2, shuffle = False)


# In[49]:


y_pred = lstm.predict(X_test)


# In[50]:


len(y_test)


# In[51]:


len(y_pred)


# In[61]:


plt.plot(y_date_index[2:len(y_date_index)], 
         y_pred, label='Forecast', color='blue')
plt.plot(y_date_index[2:len(y_date_index)], 
         y_test, label='Actual', color='green')
plt.xlabel('Year')
plt.ylabel('Close Price')
plt.title('Prediction of '+symbol_data[1]+ ' Price - LSTM Model #3')
plt.legend()
plt.show()

rmse = mean_squared_error(y_test, y_pred, squared=False)
mape = mean_absolute_percentage_error(y_test, y_pred)
print("RSME: ", rmse)
print("MAPE: ", mape)


# In[55]:


#SIMPLE MOVING AVERAGE

train_split = 0.8
split_idx = int(np.ceil(len(X_feat)*train_split))
train = X_feat[['Close']].iloc[:split_idx]
test = X_feat[['Close']].iloc[split_idx:]

test_pred = np.array([train.rolling(10).mean().iloc[-1]]*len(test)).reshape((-1,1))

plt.figure(figsize=(10,5))
plt.plot(test.index, test)
plt.plot(test.index, test_pred)
plt.xlabel('Year')
plt.ylabel('Close Price')
plt.title('Simple Moving Average - ETH')
plt.show()

print('Test RMSE: %.3f' % mean_squared_error(test, test_pred, squared=False))
print('Test MAPE: %.3f' % mean_absolute_percentage_error(test, test_pred))


# In[60]:


#Exponential Moving Average

import warnings
warnings.filterwarnings('ignore')
    
from statsmodels.tsa.api import SimpleExpSmoothing

X = X_feat[['Close']].values
train_split = 0.8
split_idx = int(np.ceil(len(X)*train_split))
train = X[:split_idx]
test = X[split_idx:]
test_concat = np.array([]).reshape((0,1))

for i in range(len(test)):
    train_fit = np.concatenate((train, np.asarray(test_concat)))
    fit = SimpleExpSmoothing(np.asarray(train_fit)).fit(smoothing_level=0.2)
    test_pred = fit.forecast(1)
    test_concat = np.concatenate((np.asarray(test_concat), test_pred.reshape((-1,1))))

plt.figure(figsize=(10,5))
plt.plot(test)
plt.plot(test_concat)
plt.xlabel('Window')
plt.ylabel('Close Price')
plt.title('Exponential Moving Average - ETH')
plt.show()

print('Test RMSE: %.3f' % mean_squared_error(test, test_concat, squared=False))
print('Test MAPE: %.3f' % mean_absolute_percentage_error(test, test_concat))


# In[59]:





# In[ ]:




