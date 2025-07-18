import pandas as pd
import streamlit as st
import sqlite3

@st.cache_data 
def load_data():
    connection = sqlite3.connect("usgsearthquakes.db",detect_types=sqlite3.PARSE_DECLTYPES) #https://stackoverflow.com/a/1830499
    df = pd.read_sql_query("SELECT mag, time, sig, nst, dmin, rms, gap, magType, id, longitude, latitude, depth FROM earthquakes;", connection)
    connection.close()
    return df

pg = st.navigation([st.Page("Prediction.py"),st.Page("Map.py"),st.Page("Dataset.py")])
pg.run()

