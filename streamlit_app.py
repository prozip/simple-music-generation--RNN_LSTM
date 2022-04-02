import streamlit as st
import subprocess
import tensorflow as tf

st.title('Music Generation with RNN/LSTM')
st.write(tf.__version__)
subprocess.run(["python", "predict.py"])