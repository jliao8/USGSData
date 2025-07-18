import streamlit as st
from Home import load_data

df = load_data()
st.title("Earthquake Dataset")
st.dataframe(df)

