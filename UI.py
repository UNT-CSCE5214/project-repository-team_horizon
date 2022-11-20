import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data

from keras.models import load_model

import streamlit as st

start = '2010-10-10'
end = '2022-10-10'

st.title("Build and Deploy Stock Market App Using Streamlit")

user_input = st.text_input('Enter','AAPL')

df = data.DataReader(user_input,'yahoo',start,end)


st.subheader('Data from ')
st.write(df.describe())

st.subheader('CLosing Price Vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)


#Adding to print Visualization for data with 100dma
st.subheader('CLosing Price Vs Time Chart vs 100')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

#Adding to print Visualization for data with 200dma


st.subheader('CLosing Price Vs Time Chart vs 200')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)


data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)

