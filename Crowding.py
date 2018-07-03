# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 17:15:15 2018

@author: mljmo
"""
import os
import numpy as np
import pandas as pd
from shapely import geometry
import shapely
import geopandas as gpd
import networkx as nx
import random

def getRoadSegments(df, Roads, buffersize=0.02):    
    Buffer = shapely.ops.cascaded_union(df['geometry'].apply(lambda x: x.buffer(buffersize)))
    return Roads[Roads["geometry"].intersects(Buffer)]

def crowding(df, Roads, buffersize=0.02):
    FakePoints = []
    for road in getRoadSegments(df, Roads, buffersize)['geometry']:
        for point in list(road.coords):
            FakePoints.append(geometry.Point(point))
            
    for point in random.sample(FakePoints,500):
        df = df.append({'geometry':point}, ignore_index=True)
    return df
    

Path = r"C:\Users\mljmo\OneDrive\GIMA\Thesis\Data\joinedtracks"
os.chdir(Path)    

df = pd.read_csv(r"joinedtracks/joinedTracks.csv")
df = gpd.GeoDataFrame(df)
df['geometry'] = df.apply(lambda row: geometry.Point(row["X"],row["Y"]), axis=1)

Roads = gpd.read_file(r"FietsersBondnetwerk_NL/links.shp")
Roads = Roads[Roads["provincie"]=="Noord-Brabant"]
Roads.crs = {'init':'epsg:28992'}
Roads = Roads.to_crs({'init': 'epsg:4326'})



track = df.head(500)
df = track.copy()
df = crowding(df, Roads)
df.plot(ax=base, c='red', marker=',')
