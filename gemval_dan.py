 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import statsmodels.formula.api as smf
import statsmodels.api as sm
from pandas.plotting import register_matplotlib_converters
from tabulate import tabulate
register_matplotlib_converters()

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from keras.callbacks import EarlyStopping

gemval=pd.read_csv("C:/Users/Vijaya/gemval_index.csv",parse_dates = ['date'], index_col = ['date'])
#gemval['value'] = gemval['value'].astype(int)
#gemval.dtypes
#Creating Train set
gemval_train_6m    = gemval[0:188]
#Creating Test set
gemval_test_6m     = gemval[188:194]
gemval_train_6m.dtypes
gemval_train_6m.describe()
gemval_train_6m.skew()
gemval_train_6m.kurt()


plt.xlabel('date')
plt.ylabel('value')
plt.plot(gemval_train_6m)
#creating log for traina nd test
gemval_train_6m   = np.log(gemval_train_6m)
gemval_test_6m     =np.log(gemval_test_6m) 
#Plotting Log transform Time Series
plt.xlabel('date')
plt.ylabel('value')
plt.plot(gemval_train_6m)
# I will use log transormed data




###  LSTM 6 months ###
#__________________#
from sklearn.preprocessing import StandardScaler
scaler  = StandardScaler()
lstm_gemval_6m = scaler.fit_transform(gemval_train_6m)

X_train = []
y_train = []
for i in range(6, len(gemval_train_6m)-6):
    X_train.append(lstm_gemval_6m[i-6:i, 0])
    y_train.append(lstm_gemval_6m[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

model = Sequential()
model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))  
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
model.add(Dense(units = 1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error')

model.fit(X_train, y_train, epochs = 100, batch_size = 32)

dataset_train_gemval_6m = gemval_train_6m.iloc[:182]
dataset_test_gemval_6m = gemval_train_6m.iloc[182:]
dataset_total_gemval_6m = pd.concat((dataset_train_gemval_6m, dataset_test_gemval_6m), axis = 0)
inputs_gemval_6m = dataset_total_gemval_6m[len(dataset_total_gemval_6m) - len(dataset_test_gemval_6m) - 6:].values
inputs_gemval_6m = inputs_gemval_6m.reshape(-1,1)
inputs_gemval_6m = scaler.transform(inputs_gemval_6m)
X_test_gemval_6m = []
for i in range(6, 12):
    X_test_gemval_6m.append(inputs_gemval_6m[i-6:i, 0])
X_test_gemval_6m = np.array(X_test_gemval_6m)
X_test_gemval_6m = np.reshape(X_test_gemval_6m, (X_test_gemval_6m.shape[0], X_test_gemval_6m.shape[1], 1))
print(X_test_gemval_6m.shape)

pred_gemval_6m = model.predict(X_test_gemval_6m)
pred_gemval_6m = scaler.inverse_transform(pred_gemval_6m)

plt.plot(dataset_test_gemval_6m, color='black', label="actual")
plt.plot(dataset_test_gemval_6m.index, pred_gemval_6m , color= 'green', label="predicted")
plt.title("LSTM")
plt.xlabel("date")
plt.ylabel("value")
plt.legend()
plt.show()

rmspe_gemval_lstm_6m = np.sqrt(mean_squared_error(gemval_test_6m['value'], pred_gemval_6m))
rmspe_gemval_lstm_6m





### Exponenatial 6 months ###
#_________________________#
gemval_model_6m = ExponentialSmoothing(gemval_train_6m, trend='add', seasonal=None)
gemval_model2_6m = ExponentialSmoothing(gemval_train_6m, trend='add', seasonal=None, damped=True)

gemval_fit1_6m = gemval_model_6m.fit()
gemval_fit2_6m = gemval_model2_6m.fit()
gemval_pred1_6m = gemval_fit1_6m.forecast(6)
gemval_pred2_6m = gemval_fit2_6m.forecast(6)
gemval_fit1_6m.summary()
gemval_pred1_6m

plt.rcParams["figure.figsize"] = [16,9]
plt.plot(gemval_train_6m, label='Train')
plt.plot(gemval_test_6m.index, gemval_pred1_6m, label='Exponential Smoothing ')
plt.legend(loc='best')
plt.show()

plt.rcParams["figure.figsize"] = [16,9]
plt.plot(gemval_train_6m, label='Train')
plt.plot(gemval_test_6m.index, gemval_pred2_6m, label='Exponential Smoothing ')
plt.legend(loc='best')
plt.show()

gemval_test_6m['pred_1']  =  gemval_pred1_6m.values
rmspe_gemval_Exp1_6m = np.sqrt(np.sum(gemval_test_6m.iloc[:, 0].subtract(gemval_test_6m.iloc[:, 1])**2)/6)
gemval_test_6m['pred_2']  =  gemval_pred2_6m.values
rmspe_gemval_Exp2_6m = np.sqrt(np.sum(gemval_test_6m.iloc[:, 1].subtract(gemval_test_6m.iloc[:, 2])**2)/6)
rmspe_gemval_Exp1_6m
rmspe_gemval_Exp2_6m









#Arima model (1 0 0) 6 months gemval
plot_acf(gemval_train_6m)
gemval_train_6m =gemval_train_6m.reset_index()
y_gemval_6m = gemval_train_6m['value']
x_gemval_6m = range(0, 188)
gemval_train_6m.columns.values
gemval_6m = smf.ols('y_gemval_6m ~ x_gemval_6m', data =gemval_train_6m)
gemval_6m=  gemval_6m.fit()
gemval_6m.summary()
plt.plot(gemval_6m.fittedvalues)
plt.plot(y_gemval_6m)

gemval_stationary_6m = y_gemval_6m - gemval_6m.fittedvalues
plt.plot(gemval_stationary_6m)
plot_acf(gemval_stationary_6m)
plot_pacf(gemval_stationary_6m) #AR1 should be good, although number 14 is a bit worring.

gemval_ar1_6m = ARIMA(y_gemval_6m, order = (1, 0, 0))
gemval_ar1_6m = gemval_ar1_6m.fit()

fig = plt.figure() 
ax1 = fig.add_subplot(3, 1, 1) # number of row and column + position
ax2 = fig.add_subplot(3, 1, 2)
ax3 = fig.add_subplot(3, 1, 3)
ax1.plot(gemval_ar1_6m.resid)
plot_acf(gemval_ar1_6m.resid, ax = ax2)
plot_pacf(gemval_ar1_6m.resid, ax = ax3)
fig.tight_layout()
plt.show() #all the short term correlation and random variation seem to be removed

gemval_ar1_pred_6m = gemval_ar1_6m.predict(start = 188, end = 194)
plt.plot(y_gemval_6m)
plt.plot(gemval_ar1_pred_6m)
# make the predictions for 1

gemval_ar1_2_6m = SARIMAX(y_gemval_6m, order = (1, 0, 0))
gemval_ar1_2_6m = gemval_ar1_2_6m.fit()
gemval_ar1_2_6m.summary() 

y_pred_gemval_6m = gemval_ar1_2_6m.get_forecast(len(gemval_test_6m.index)+1)
y_pred_df_gemval_6m = y_pred_gemval_6m.conf_int(alpha = 0.05) 
y_pred_df_gemval_6m["Predictions"] = gemval_ar1_2_6m.predict(start = y_pred_df_gemval_6m.index[0], end = y_pred_df_gemval_6m.index[-1])
#conf_df = pd.concat([test['MI'],predictions_int.predicted_mean, predictions_int.conf_int()], axis = 1)
y_pred_gemval_6m.conf_int()
#conf_df.head()
fig = plt.figure(figsize = (16,8))
ax1 = fig.add_subplot(1, 1, 1)
plt.plot(y_gemval_6m)
plt.plot(y_pred_df_gemval_6m["Predictions"],label='predicted')
y_pred_df_gemval_6m.head()
plt.plot(y_pred_df_gemval_6m['lower value'], linestyle = '--', color = 'red', linewidth = 0.5,label='lower ci')
plt.plot(y_pred_df_gemval_6m['upper value'], linestyle = '--', color = 'red', linewidth = 0.5,label='upper ci')
plt.fill_between(y_pred_df_gemval_6m["Predictions"].index.values,
                 y_pred_df_gemval_6m['upper value'],
                 y_pred_df_gemval_6m['lower value'],
                 color = 'grey', alpha = 0.2)
plt.legend(loc = 'lower left', fontsize = 12)
plt.show()

gemval_ar_pred_6m = pd.DataFrame(gemval_ar1_pred_6m)
gemval_test_6m['pred_1'] = gemval_ar_pred_6m.values[1:7]
rmspe1_gemval_1_6m = np.sqrt(np.sum(gemval_test_6m.iloc[:, 0].subtract(gemval_test_6m.iloc[:, 1])**2)/6)

y_pred_df_gemval_6m.columns.values
pred_2_6m = pd.DataFrame(y_pred_df_gemval_6m['Predictions'])
gemval_test_6m['pred_2_w'] = pred_2_6m.values[1:7]
rmspe1_gemval_2_6m = np.sqrt(np.sum(gemval_test_6m.iloc[:, 0].subtract(gemval_test_6m.iloc[:, 2])**2)/6)


#Arima(1,0,1)
gemval_ar1_6m2 = ARIMA(y_gemval_6m, order = (1, 0, 1))
gemval_ar1_6m2 = gemval_ar1_6m2.fit()

fig = plt.figure() 
ax1 = fig.add_subplot(3, 1, 1) # number of row and column + position
ax2 = fig.add_subplot(3, 1, 2)
ax3 = fig.add_subplot(3, 1, 3)
ax1.plot(gemval_ar1_6m2.resid)
plot_acf(gemval_ar1_6m2.resid, ax = ax2)
plot_pacf(gemval_ar1_6m2.resid, ax = ax3)
fig.tight_layout()
plt.show()#all the short term correlation and random variation seem to be removed

gemval_ar1_pred_6m2 = gemval_ar1_6m2.predict(start = 188, end = 194)
plt.plot(y_gemval_6m)
plt.plot(gemval_ar1_pred_6m2)

gemval_ar1_2_6m2 = SARIMAX(y_gemval_6m, order = (1, 0, 1)) 
gemval_ar1_2_6m2 = gemval_ar1_2_6m2.fit()
y_pred_gemval_6m2 = gemval_ar1_2_6m2.get_forecast(len(gemval_test_6m.index)+1)
y_pred_df_gemval_6m2 = y_pred_gemval_6m2.conf_int(alpha = 0.05) 
y_pred_df_gemval_6m2["Predictions"] = gemval_ar1_2_6m2.predict(start = y_pred_df_gemval_6m2.index[0], end = y_pred_df_gemval_6m2.index[-1])

fig = plt.figure(figsize = (16,8))
ax1 = fig.add_subplot(1, 1, 1)
plt.plot(y_gemval_6m)
plt.plot(y_pred_df_gemval_6m2["Predictions"],color = 'red',label='predicted')
y_pred_df_gemval_6m2.head()
plt.plot(y_pred_df_gemval_6m2['lower value'], linestyle = '--', color = 'red', linewidth = 0.5, label='lower ci')
plt.plot(y_pred_df_gemval_6m2['upper value'], linestyle = '--', color = 'red', linewidth = 0.5, label='upper ci')
plt.fill_between(y_pred_df_gemval_6m2["Predictions"].index.values,
                 y_pred_df_gemval_6m2['lower value'], 
                 y_pred_df_gemval_6m2['upper value'], 
                 color = 'grey', alpha = 0.2)
plt.legend(loc = 'best')
plt.show()

gemval_ar_pred_6m2 = pd.DataFrame(gemval_ar1_pred_6m2)
gemval_test_6m['pred_1'] = gemval_ar_pred_6m2.values[1:7]
rmspe2_gemval_1_6m = np.sqrt(np.sum(gemval_test_6m.iloc[:, 0].subtract(gemval_test_6m.iloc[:, 1])**2)/6)

y_pred_df_gemval_6m2.columns.values
pred_2_6m2 = pd.DataFrame(y_pred_df_gemval_6m2['Predictions'])
gemval_test_6m['pred_2_w'] = pred_2_6m2.values[1:7]
rmspe2_gemval_2_6m = np.sqrt(np.sum(gemval_test_6m.iloc[:, 0].subtract(gemval_test_6m.iloc[:, 2])**2)/6)






#####  1 year EXPONENTIAL SMOOTHING  gemval ######
#------------------------------------------------------------#
#1 year
gemval_Train_1y    = gemval[0:182]
gemval_Test_1y     = gemval[182:194]
gemval_Train_1y    = np.log(gemval_Train_1y)
gemval_Test_1y    = np.log(gemval_Test_1y)

plt.xlabel('date')
plt.ylabel('value')
plt.plot(gemval_Train_1y)

gemval_model_1y = ExponentialSmoothing(gemval_Train_1y, trend='add', seasonal=None)
gemval_model2_1y = ExponentialSmoothing(gemval_Train_1y, trend='add', seasonal=None, damped=True)

gemval_fit1_1y = gemval_model_1y.fit()
gemval_fit2_1y = gemval_model2_1y.fit()
gemval_pred1_1y = gemval_fit1_1y.forecast(12)
gemval_pred2_1y = gemval_fit2_1y.forecast(12)
gemval_fit1_1y.summary()
gemval_pred1_1y

plt.rcParams["figure.figsize"] = [16,9]
plt.plot(gemval_Train_1y, label='Train')
plt.plot(gemval_Test_1y.index, gemval_pred1_1y, label='Exponential Smoothing ')
plt.legend(loc='best')
plt.show()

plt.rcParams["figure.figsize"] = [16,9]
plt.plot(gemval_Train_1y, label='Train')
plt.plot(gemval_Test_1y.index, gemval_pred2_1y, label='Exponential Smoothing ')
plt.legend(loc='best')
plt.show()

gemval_Test_1y['pred_1']  =  gemval_pred1_1y.values
rmspe_gemval_Exp1_1y = np.sqrt(np.sum(gemval_Test_1y.iloc[:, 0].subtract(gemval_Test_1y.iloc[:, 1])**2)/12)
gemval_Test_1y['pred_2']  =  gemval_pred2_1y.values
rmspe_gemval_Exp2_1y = np.sqrt(np.sum(gemval_Test_1y.iloc[:, 1].subtract(gemval_Test_1y.iloc[:, 2])**2)/12)


### LSTM 1 year ###
#------------------#
lstm_gemval_1y = scaler.fit_transform(gemval_Train_1y)

X_train = []
y_train = []
for i in range(12, len(gemval_Train_1y)-12):
    X_train.append(lstm_gemval_1y[i-12:i, 0])
    y_train.append(lstm_gemval_1y[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

model = Sequential()
model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))  
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
model.add(Dense(units = 1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error')

model.fit(X_train, y_train, epochs = 100, batch_size = 32)

dataset_train_gemval_1y = gemval_Train_1y.iloc[:170]
dataset_test_gemval_1y = gemval_Train_1y.iloc[170:]
dataset_total_gemval_1y = pd.concat((dataset_train_gemval_1y, dataset_test_gemval_1y), axis = 0)
inputs_gemval_1y = dataset_total_gemval_1y[len(dataset_total_gemval_1y) - len(dataset_test_gemval_1y) - 12:].values
inputs_gemval_1y = inputs_gemval_1y.reshape(-1,1)
inputs_gemval_1y = scaler.transform(inputs_gemval_1y)
X_test_gemval_1y = []
for i in range(12, 24):
    X_test_gemval_1y.append(inputs_gemval_1y[i-12:i, 0])
X_test_gemval_1y = np.array(X_test_gemval_1y)
X_test_gemval_1y = np.reshape(X_test_gemval_1y, (X_test_gemval_1y.shape[0], X_test_gemval_1y.shape[1], 1))
print(X_test_gemval_1y.shape)

pred_gemval_1y = model.predict(X_test_gemval_1y)
pred_gemval_1y = scaler.inverse_transform(pred_gemval_1y)

plt.plot(dataset_test_gemval_1y, color='black', label="actual")
plt.plot(dataset_test_gemval_1y.index, pred_gemval_1y , color= 'green', label="predicted")
plt.title("LSTM")
plt.xlabel("date")
plt.ylabel("value")
plt.legend()
plt.show()

rmspe_gemval_lstm_1y = np.sqrt(mean_squared_error(gemval_Test_1y['value'], pred_gemval_1y))
rmspe_gemval_lstm_1y


#1 year ARIMA
plot_acf(gemval_Train_1y)

gemval_Train_1y = gemval_Train_1y.reset_index()
y_gemval_1y = gemval_Train_1y['value']
x_gemval_1y = range(0, 182)
gemval_lm_1y = smf.ols('y_gemval_1y ~ x_gemval_1y', data = gemval_Train_1y)
gemval_lm_1y = gemval_lm_1y.fit()
gemval_lm_1y.summary()
plt.plot(gemval_lm_1y.fittedvalues)
plt.plot(y_gemval_1y)

gemval_stationary_1y = y_gemval_1y - gemval_lm_1y.fittedvalues
plt.plot(gemval_stationary_1y)
plot_acf(gemval_stationary_1y)
plot_pacf(gemval_stationary_1y)

gemval_ar_1y = ARIMA(y_gemval_1y, order = (1, 0, 0))
gemval_ar_1y = gemval_ar_1y.fit()

fig = plt.figure() 
ax1 = fig.add_subplot(3, 1, 1) # number of row and column + position
ax2 = fig.add_subplot(3, 1, 2)
ax3 = fig.add_subplot(3, 1, 3)
ax1.plot(gemval_ar_1y.resid)
plot_acf(gemval_ar_1y.resid, ax = ax2)
plot_pacf(gemval_ar_1y.resid, ax = ax3)
fig.tight_layout()
plt.show()

gemval_ar_pred_1y = gemval_ar_1y.predict(start = 182, end = 194)

plt.plot(y_gemval_1y)
plt.plot(gemval_ar_pred_1y)
gemval_ar_2_1y = SARIMAX(y_gemval_1y, order = (1, 0, 0))

gemval_ar_2_1y = gemval_ar_2_1y.fit()
y_pred_gemval_1y = gemval_ar_2_1y.get_forecast(len(gemval_Test_1y.index)+1)
y_pred_df_gemval_1y = y_pred_gemval_1y.conf_int(alpha = 0.05) 
y_pred_df_gemval_1y["Predictions"] = gemval_ar_2_1y.predict(start = y_pred_df_gemval_1y.index[0], end = y_pred_df_gemval_1y.index[-1])

fig = plt.figure(figsize = (16,8))
ax1 = fig.add_subplot(1, 1, 1)
plt.plot(y_gemval_1y)
plt.plot(y_pred_df_gemval_1y["Predictions"],color = 'red',label='predicted')
y_pred_df_gemval_1y.head()
plt.plot(y_pred_df_gemval_1y['lower value'], linestyle = '--', color = 'red', linewidth = 0.5, label='lower ci')
plt.plot(y_pred_df_gemval_1y['upper value'], linestyle = '--', color = 'red', linewidth = 0.5, label='upper ci')
plt.fill_between(y_pred_df_gemval_1y["Predictions"].index.values,
                 y_pred_df_gemval_1y['lower value'], 
                 y_pred_df_gemval_1y['upper value'], 
                 color = 'grey', alpha = 0.2)
plt.legend(loc = 'best')
plt.show()

gemval_ar_pred_1y = pd.DataFrame(gemval_ar_pred_1y)
gemval_Test_1y['pred_1'] = gemval_ar_pred_1y.values[1:13]
rmspe_gemval_1_1y = np.sqrt(np.sum(gemval_Test_1y.iloc[:, 0].subtract(gemval_Test_1y.iloc[:, 1])**2)/12)

pred_2_1y = pd.DataFrame(y_pred_df_gemval_1y['Predictions'])
gemval_Test_1y['pred_2_1y'] = pred_2_1y.values[1:13]
rmspe__gemval_2_1y = np.sqrt(np.sum(gemval_Test_1y.iloc[:, 0].subtract(gemval_Test_1y.iloc[:, 2])**2)/12)


#Arima(1 0 1)
gemval_ar1_1y2 = ARIMA(y_gemval_1y, order = (1, 0, 1))
gemval_ar1_1y2 = gemval_ar1_1y2.fit()

fig = plt.figure() 
ax1 = fig.add_subplot(3, 1, 1) # number of row and column + position
ax2 = fig.add_subplot(3, 1, 2)
ax3 = fig.add_subplot(3, 1, 3)
ax1.plot(gemval_ar1_1y2.resid)
plot_acf(gemval_ar1_1y2.resid, ax = ax2)
plot_pacf(gemval_ar1_1y2.resid, ax = ax3)
fig.tight_layout()
plt.show()#all the short term correlation and random variation seem to be removed

gemval_ar1_pred_1y2 = gemval_ar1_1y2.predict(start = 182, end = 194)
plt.plot(y_gemval_1y)
plt.plot(gemval_ar1_pred_1y2)

gemval_ar1_2_1y2 = SARIMAX(y_gemval_1y, order = (1, 0, 1)) 
gemval_ar1_2_1y2 = gemval_ar1_2_1y2.fit()
y_pred_gemval_1y2 = gemval_ar1_2_1y2.get_forecast(len(gemval_Test_1y.index)+1)
y_pred_df_gemval_1y2 = y_pred_gemval_1y2.conf_int(alpha = 0.05) 
y_pred_df_gemval_1y2["Predictions"] = gemval_ar1_2_1y2.predict(start = y_pred_df_gemval_1y2.index[0], end = y_pred_df_gemval_1y2.index[-1])

fig = plt.figure(figsize = (16,8))
ax1 = fig.add_subplot(1, 1, 1)
plt.plot(y_gemval_1y)
plt.plot(y_pred_df_gemval_1y2["Predictions"],color = 'red',label='predicted')
y_pred_df_gemval_1y2.head()
plt.plot(y_pred_df_gemval_1y2['lower value'], linestyle = '--', color = 'red', linewidth = 0.5, label='lower ci')
plt.plot(y_pred_df_gemval_1y2['upper value'], linestyle = '--', color = 'red', linewidth = 0.5, label='upper ci')
plt.fill_between(y_pred_df_gemval_1y2["Predictions"].index.values,
                 y_pred_df_gemval_1y2['lower value'], 
                 y_pred_df_gemval_1y2['upper value'], 
                 color = 'grey', alpha = 0.2)
plt.legend(loc = 'best')
plt.show()

gemval_ar_pred_1y2 = pd.DataFrame(gemval_ar1_pred_1y2)
gemval_Test_1y['pred_1'] = gemval_ar_pred_1y2.values[1:13]
rmspe2_gemval_1_1y = np.sqrt(np.sum(gemval_Test_1y.iloc[:, 0].subtract(gemval_Test_1y.iloc[:, 1])**2)/12)

y_pred_df_gemval_1y2.columns.values
pred_2_1y2 = pd.DataFrame(y_pred_df_gemval_1y2['Predictions'])
gemval_Test_1y['pred_2_m'] = pred_2_1y2.values[1:13]
rmspe2_gemval_2_1y = np.sqrt(np.sum(gemval_Test_1y.iloc[:, 0].subtract(gemval_Test_1y.iloc[:, 2])**2)/12)





### 2 years Exponential smoothing ###
#------------------------------------#
gemval_Train_2y    = gemval[0:170]
gemval_Test_2y    = gemval[170:194]
gemval_Train_2y    = np.log(gemval_Train_2y)
gemval_Test_2y    = np.log(gemval_Test_2y)
gemval_model_2y = ExponentialSmoothing(gemval_Train_2y, trend='add', seasonal=None)
gemval_model2_2y = ExponentialSmoothing(gemval_Train_2y, trend='add', seasonal=None, damped=True)

gemval_fit1_2y = gemval_model_2y.fit()
gemval_fit2_2y = gemval_model2_2y.fit()
gemval_pred1_2y = gemval_fit1_2y.forecast(24)
gemval_pred2_2y = gemval_fit2_2y.forecast(24)

plt.rcParams["figure.figsize"] = [16,9]
plt.plot(gemval_Train_2y, label='Train')
plt.plot(gemval_Test_2y.index, gemval_pred1_2y, label='predicted line for Exponential Smoothing')
plt.legend(loc='best')
plt.show()

plt.rcParams["figure.figsize"] = [16,9]
plt.plot(gemval_Train_2y, label='Train')
plt.plot(gemval_Test_2y.index, gemval_pred2_2y, label='predicted line for Exponential Smoothing')
plt.legend(loc='best')
plt.show()

gemval_Test_2y['pred_1']  =  gemval_pred1_2y.values
rmspe_gemval_Exp1_2y = np.sqrt(np.sum(gemval_Test_2y.iloc[:, 0].subtract(gemval_Test_2y.iloc[:, 1])**2)/24)
gemval_Test_2y['pred_2']  =  gemval_pred2_2y.values
rmspe_gemval_Exp2_2y = np.sqrt(np.sum(gemval_Test_2y.iloc[:, 0].subtract(gemval_Test_2y.iloc[:, 2])**2)/24)





### LSTM 2 years ###
#-------------------#
lstm_gemval_2y = scaler.fit_transform(gemval_Train_2y)

X_train = []
y_train = []
for i in range(24, len(gemval_Train_2y)-24):
    X_train.append(lstm_gemval_2y[i-24:i, 0])
    y_train.append(lstm_gemval_2y[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

model = Sequential()
model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))  
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
model.add(Dense(units = 1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error')

model.fit(X_train, y_train, epochs = 100, batch_size = 32)

dataset_train_gemval_2y = gemval_Train_2y.iloc[:146]
dataset_test_gemval_2y = gemval_Train_2y.iloc[146:]
dataset_total_gemval_2y = pd.concat((dataset_train_gemval_2y, dataset_test_gemval_2y), axis = 0)
inputs_gemval_2y = dataset_total_gemval_2y[len(dataset_total_gemval_2y) - len(dataset_test_gemval_2y) - 24:].values
inputs_gemval_2y = inputs_gemval_2y.reshape(-1,1)
inputs_gemval_2y = scaler.transform(inputs_gemval_2y)
X_test_gemval_2y = []
for i in range(24, 48):
    X_test_gemval_2y.append(inputs_gemval_2y[i-24:i, 0])
X_test_gemval_2y = np.array(X_test_gemval_2y)
X_test_gemval_2y = np.reshape(X_test_gemval_2y, (X_test_gemval_2y.shape[0], X_test_gemval_2y.shape[1], 1))
print(X_test_gemval_2y.shape)

pred_gemval_2y = model.predict(X_test_gemval_2y)
pred_gemval_2y = scaler.inverse_transform(pred_gemval_2y)

plt.plot(dataset_test_gemval_2y, color='black', label="actual")
plt.plot(dataset_test_gemval_2y.index, pred_gemval_2y , color= 'green', label="predicted")
plt.title("LSTM")
plt.xlabel("date")
plt.ylabel("value")
plt.legend()
plt.show()

rmspe_gemval_lstm_2y = np.sqrt(mean_squared_error(gemval_Test_2y['value'], pred_gemval_2y))
rmspe_gemval_lstm_2y





### Arima(2 years) ###
#---------------------#
gemval_Train_2y = gemval_Train_2y.reset_index()
y_gemval_2y = gemval_Train_2y['value']
x_gemval_2y = range(0, 170)
gemval_Train_2y.columns.values
gemval_lm_2y = smf.ols('y_gemval_2y ~ x_gemval_2y', data = gemval_Train_2y)
gemval_lm_2y = gemval_lm_2y.fit()
gemval_lm_2y.summary()
plt.plot(gemval_lm_2y.fittedvalues)
plt.plot(y_gemval_2y)

gemval_stationary_2y = y_gemval_2y - gemval_lm_2y.fittedvalues
plt.plot(gemval_stationary_2y)
plot_acf(gemval_stationary_2y)
plot_pacf(gemval_stationary_2y)# AR1 ok, higher numbers problematics

gemval_ar_2y = ARIMA(y_gemval_2y, order = (1, 0, 0))
gemval_ar_2y = gemval_ar_2y.fit()


fig = plt.figure() 
ax1 = fig.add_subplot(3, 1, 1) # number of row and column + position
ax2 = fig.add_subplot(3, 1, 2)
ax3 = fig.add_subplot(3, 1, 3)
ax1.plot(gemval_ar_2y.resid)
plot_acf(gemval_ar_2y.resid, ax = ax2)
plot_pacf(gemval_ar_2y.resid, ax = ax3)
fig.tight_layout()
plt.show()

gemval_ar_pred_2y2 = gemval_ar_2y.predict(start = 170, end = 194)

plt.plot(y_gemval_2y)
plt.plot(gemval_ar_pred_2y2)

gemval_ar_2_2y2 = SARIMAX(y_gemval_2y, order = (1, 0, 0))
gemval_ar_2_2y2 = gemval_ar_2_2y2.fit()
y_pred_gemval_2y2 = gemval_ar_2_2y2.get_forecast(len(gemval_Test_2y.index)+1)
y_pred_df_gemval_2y2 = y_pred_gemval_2y2.conf_int(alpha = 0.05) 
y_pred_df_gemval_2y2["Predictions"] = gemval_ar_2_2y2.predict(start = y_pred_df_gemval_2y2.index[0], end = y_pred_df_gemval_2y2.index[-1])

fig = plt.figure(figsize = (16,8))
ax1 = fig.add_subplot(1, 1, 1)
plt.plot(y_gemval_2y)
plt.plot(y_pred_df_gemval_2y2["Predictions"],color = 'red',label='predicted')
plt.plot(y_pred_df_gemval_2y2['lower value'], linestyle = '--', color = 'red', linewidth = 0.5, label='lower ci')
plt.plot(y_pred_df_gemval_2y2['upper value'], linestyle = '--', color = 'red', linewidth = 0.5, label='upper ci')
plt.fill_between(y_pred_df_gemval_2y2["Predictions"].index.values,
                 y_pred_df_gemval_2y2['lower value'], 
                 y_pred_df_gemval_2y2['upper value'], 
                 color = 'grey', alpha = 0.2)
plt.legend(loc = 'best')
plt.show()

gemval_ar_pred_2y2 = pd.DataFrame(gemval_ar_pred_2y2)
gemval_Test_2y['pred_1'] = gemval_ar_pred_2y2.values[1:25]
rmspe__gemval_1_2y2 = np.sqrt(np.sum(gemval_Test_2y.iloc[:, 0].subtract(gemval_Test_2y.iloc[:, 1])**2)/24)

y_pred_df_gemval_2y2.columns.values
pred_2_2y2 = pd.DataFrame(y_pred_df_gemval_2y2['Predictions'])
gemval_Test_2y['pred_2_2y'] = pred_2_2y2.values[1:25]
rmspe_gemval_2_2y2 = np.sqrt(np.sum(gemval_Test_2y.iloc[:, 0].subtract(gemval_Test_2y.iloc[:, 2])**2)/24)




gemval_rmspe = [['ARIMA (1 0 0) - 6 months 1', rmspe1_gemval_1_6m],
                ['SARIMA (1 0 0) - 6 months 2', rmspe1_gemval_2_6m],
                ['ARIMA (1 0 1) - 6 months 1', rmspe2_gemval_1_6m],
                ['SARIMA (1 0 1) - 6 months 2', rmspe2_gemval_2_6m],
                ['ARIMA (1 0 0) - 1 year 1', rmspe_gemval_1_1y],
                ['SARIMA (1 0 0) - 1 year 2', rmspe__gemval_2_1y],
                ['ARIMA (1 0 1) - 1 year 1', rmspe2_gemval_1_1y],
                ['SARIMA (1 0 1) - 1 year 2', rmspe2_gemval_2_1y],
                ['ARIMA (1 0 0) - 2 years 1', rmspe__gemval_1_2y2],
                ['ARIMA (1 0 0) - 2 years 2', rmspe_gemval_2_2y2],
                ['LSTM - 6 months', rmspe_gemval_lstm_6m],
                ['LSTM - 1 year', rmspe_gemval_lstm_1y],
                ['LSTM - 2 years', rmspe_gemval_lstm_2y],
                ['EXPO - 6 months', rmspe_gemval_Exp1_6m],
                ['EXPO - 6 months damped', rmspe_gemval_Exp2_6m],
                ['EXPO - 1 year', rmspe_gemval_Exp1_1y],
                ['EXPO - 1 year damped', rmspe_gemval_Exp2_1y],
                ['EXPO - 2 years', rmspe_gemval_Exp1_2y],
                ['EXPO - 2 years damped', rmspe_gemval_Exp2_2y],
                ]

print(tabulate(gemval_rmspe, headers='firstrow', tablefmt='fancy_grid'))
gemval_rmspe = sorted(gemval_rmspe, key=lambda x:x[1])
gemval_rmspe = pd.DataFrame(gemval_rmspe)

plt.plot(gemval_rmspe[0], gemval_rmspe[1], '.')
plt.xticks(rotation=90)
plt.title('gemval')



