import streamlit as st
import subprocess, sys
import tensorflow as tf

st.title('Music Generation with RNN/LSTM')
st.write(tf.__version__)
subprocess.run([f"{sys.executable}", "predict.py"])
