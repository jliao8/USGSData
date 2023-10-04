import pandas as pd
import streamlit as st

@st.cache_data
def load_data():
    # the last 6 months taken from https://earthquake.usgs.gov/earthquakes/search/

    file1 = "/home/user/Project/2022-09-01 to 2022-10-01 World Earthquakes.csv"
    file2 = "/home/user/Project/2022-10-01 to 2022-11-16 World Earthquakes.csv"
    file3 = "/home/user/Project/2022-11-16 to 2023-01-05 World Earthquakes.csv"
    file4 = "/home/user/Project/2023-01-05 to 2023-03-01 World Earthquakes.csv"
    df1,df2,df3,df4 = pd.read_csv(file1),pd.read_csv(file2),pd.read_csv(file3),pd.read_csv(file4)
    df = pd.concat([df1[::-1],df2[::-1],df3[::-1],df4[::-1]],ignore_index = True)

    df.drop_duplicates(inplace=True) # dates do overlap although miniscule
    df.drop("type", axis=1, inplace=True) # redundant
    df.drop("updated", axis=1, inplace=True) # no valuable information
    df.drop("place", axis=1, inplace=True) # no need to have a reference point when long and lat are provided
    df.dropna(inplace=True) # drop all rows with a missing value

    for i, r in df.iterrows(): # accurate reviewed data
        if r["status"] == "automatic" or r["status"] == "deleted":
            df.drop(index=i, inplace=True)
    df.drop("status", axis=1, inplace=True) # redundant
    df["time"] = pd.to_datetime(df["time"]) # convert object to datetime

    netlocmag_identical = True
    for i, r in df.iterrows(): # check for redundancy in columns
        if r["locationSource"] != r["magSource"] or r["locationSource"] != r["net"]:
            netlocmag_identical = False
    if netlocmag_identical: # rename the column to combine and drop the others
        df.rename(columns={"net": "netlocmagSource"}, inplace=True) 
        df.drop("magSource", axis=1, inplace=True)
        df.drop("locationSource", axis=1, inplace=True)
    df.reset_index(drop=True, inplace = True)
    return df
df = load_data()

st.header("Map of Global Earthquakes")
# user input to change the date range of points
start_date = st.date_input("Start Date:").strftime('%Y-%m-%d')
end_date = st.date_input("End Date:").strftime('%Y-%m-%d')
df.set_index(['time'], inplace=True) 
df = df.loc[start_date:end_date]
st.map(df)
