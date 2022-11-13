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