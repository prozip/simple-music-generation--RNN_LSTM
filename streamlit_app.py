import streamlit as st
import subprocess, sys
import tensorflow as tf
import os

st.title('Music Generation with RNN/LSTM')
st.write(tf.__version__)
os.mkdir('test')
st.write(os.listdir())
subprocess.run([f"{sys.executable}", "predict.py"])
