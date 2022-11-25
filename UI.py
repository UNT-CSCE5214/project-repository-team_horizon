import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data

from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

import streamlit as st

start = '2000-01-01'
end = '2022-11-01'

st.title("SDAI - Academic Project - Team 13")

user_input = st.text_input('Enter','AAPL')

df = data.DataReader(user_input,'yahoo',start,end)
df.rename(columns = {"Date" : "date", "Open" : "open", "High" : "high", "Low" : "low","Close" : "close", "Volume" : "volume", "Adj Close" : "adj close"}, inplace = True)
#renaming the data frame columns to lower case
x_test =[]
y_test = []

st.subheader('Data from Website - Yahoo Finance')
st.write(df.describe())





data_training = pd.DataFrame(df['close'][0:int(len(df)*0.80)])
data_testing = pd.DataFrame(df['close'][int(len(df)*0.80):int(len(df))])
scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)

#Load models
model = load_model('2.h5')

past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing,ignore_index=True)
input_data = scaler.fit_transform(final_df)



for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test,y_test = np.array(x_test),np.array(y_test)

y_predicted = model.predict(x_test)

scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

st.subheader('closing price')
fig = plt.figure(figsize=(12,6))
plt.plot(df.close)
st.pyplot(fig)

st.subheader('closing price with 100ma')
ma100 = df.close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(df.close)
st.pyplot(fig)



st.subheader('closing price with 100ma and 200ma')
ma100 = df.close.rolling(100).mean()
ma200 = df.close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.close)
st.pyplot(fig)

st.subheader('Testing data and Prediction')
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test,'b', label = 'Original')
plt.plot(y_predicted,'r', label = 'Predicted')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)


predictions = model.predict(x_test)
#predictions = scaler.inverse_transform(predictions)
predictions = predictions * scale_factor



data = df.filter(['close'])
data
dataset = data.values
training_data_len = int(np.ceil( len(dataset) * .80 ))-1


train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
# Visualize the data




st.subheader('Main Visualization')
fig_main=plt.figure(figsize=(12,6))

plt.plot(train['close'], label='A')
plt.plot(valid[['close', 'Predictions']] ,label='B')
plt.legend(['Train', 'Val', 'Predictions'], loc = 'lower right')

st.pyplot(fig_main)
