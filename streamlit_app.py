import streamlit as st
import os
import tensorflow as tf

st.title('Music Generation with RNN/LSTM')
st.write(tf.__version__)
os.system('python3 predict.py')