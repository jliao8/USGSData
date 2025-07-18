import pandas as pd
import streamlit as st
from datetime import datetime,date
from Home import load_data

df = load_data()
st.header("Map of Global Earthquakes")
with st.sidebar:
    # date_input keeps date no time
    start_date = st.date_input("Start Date:",df["time"].min()) 
    end_date = st.date_input("End Date:",df["time"].max())

df = df.loc[(df["time"].dt.date >= start_date) & (df["time"].dt.date <= end_date)]
st.map(df)
