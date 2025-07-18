#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import aiohttp
import asyncio
import itertools
import sqlite3
from datetime import datetime, timedelta
from time import perf_counter
from urllib.parse import urlencode

SEARCH_LIMIT = 20000 # maximum number of events ComCat will return in one search
API_THROTTLE_LIMIT = (60*5)/500 # min seconds between requests
TIMEOUT = 60
TIMEFMT = "%Y-%m-%dT%H:%M:%S.%f"

# taken from USGS libcomcat https://code.usgs.gov/ghsc/esi/libcomcat-python
def get_time_segments(starttime, endtime):
    freq = 1000 # earthquakes per day for magnitude 0
    ndays = (endtime - starttime).days + 1
    nsegments = int(np.ceil((freq * ndays) / SEARCH_LIMIT))
    days_per_segment = int(np.ceil(ndays / nsegments))
    segments = []
    startseg = starttime
    endseg = starttime
    
    while startseg <= endtime:
        endseg = startseg + timedelta(days_per_segment)
        if endseg > endtime:
            endseg = endtime
        segments.append((startseg, endseg))
        startseg += timedelta(days=days_per_segment, microseconds=1)
    return segments
    
async def search(url, session, detail=False): 
    try:
        async with session.get(url, timeout=TIMEOUT) as response:
            jdict = await response.json()
            if detail:
                return jdict 
            return jdict["features"]
    except asyncio.TimeoutError as terr:
        print("timeout %s %s" % (url, terr))
    except aiohttp.ContentTypeError as cerr: # 429 too many requests
        print("response %s %s" % (response, cerr))    

async def get_data(start_time, end_time):
    start = perf_counter()
    segments = get_time_segments(start_time, end_time)
    iseg = 0
    async with aiohttp.ClientSession() as session:
        summary_tasks = []
        for stime, etime in segments:
            iseg += 1
            parameters = {"format":"geojson", "starttime":stime.strftime(TIMEFMT), "endtime":etime.strftime(TIMEFMT), 
                          "limit":20000, "minmagnitude":0, 'eventtype':'earthquake'}
            url = "https://earthquake.usgs.gov/fdsnws/event/1/query?" + urlencode(parameters)
            summary_tasks.append(asyncio.create_task(search(url, session)))
            print("Searching summary %i: %s to %s" % (iseg, stime, etime))
            # there is a throttle on the number of API requests that can be made (500 in 5 minutes.) 
            await asyncio.sleep(API_THROTTLE_LIMIT)
        summary_results = await asyncio.gather(*summary_tasks)
        # https://datascienceparichay.com/article/python-flatten-a-list-of-lists-to-a-single-list/
        summary_chain = list(itertools.chain(*summary_results)) 
        print(f"summary timer {perf_counter()-start}")
        return summary_chain

# https://expertbeacon.com/how-to-flatten-a-dictionary-in-python-an-expert-guide/
def flatten_gen(d, parent_key='',ignoreList=["type","title","place","tz","felt","cdi","mmi","alert","tsunami",
                                             "updated","status","types","net","code","ids","sources"]): 
    for key, value in d.items():
        k = f'{parent_key}_{key}' if parent_key else key
        if key in ignoreList: continue
        if isinstance(value, dict):
            yield from flatten_gen(value, parent_key=k)  
        else:
            yield k, value
        
async def process_data(start_time, end_time):
    data = await get_data(start_time, end_time) 
    flat_earthquakes = []
    for earthquake in data:
        flat_earthquakes.append(dict(flatten_gen(earthquake)))
    df = pd.DataFrame(flat_earthquakes)
    df.dropna(axis=1, how="all") # gets rid of empty columns
    df["properties_time"] = pd.to_datetime(df["properties_time"], unit="ms") # timestamp to datetime
    df["longitude"], df["latitude"], df["depth"] = zip(*list(df["geometry_coordinates"].values)) # split coordinates
    df.drop("geometry_coordinates", axis=1, inplace=True)
    df.rename(columns={c: c.replace("properties_", "") for c in df.columns}, inplace=True) # https://stackoverflow.com/a/47054585
    
    connection = sqlite3.connect("usgsearthquakes.db")
    df.to_sql(name="earthquakes", con=connection, if_exists="replace", index=False)
    connection.close()
    
asyncio.run(process_data(datetime(2000, 1, 1),datetime(2025, 1, 1))) # input date range

