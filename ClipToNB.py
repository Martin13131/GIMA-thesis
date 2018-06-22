# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 17:14:39 2018

@author: mljmo
"""

import os
import numpy as np
import pandas as pd
from shapely import geometry
import geopandas as gpd


Path = r"C:\Users\mljmo\OneDrive\GIMA\Thesis\Data"
os.chdir(Path)

df = pd.read_csv(r"joinedtracks\joinedTracks.csv")

NBrabant = gpd.read_file("NBrabant.shp")
NBrabant["Dissolve"] = True
NBrabant = NBrabant.dissolve(by="Dissolve").buffer(500).to_crs({'init': 'epsg:4326'})
Data = []
for track, trackdf in df.groupby("track"):
    x, y = trackdf["X"].values, trackdf["Y"].values
    points = geometry.MultiPoint(tuple(zip(x,y)))
    if len(points) > 1 and NBrabant.contains(points).all():
        lines = geometry.LineString(points)
    else:
        continue
    Data.append([track, points,lines])

gdf = gpd.GeoDataFrame(Data, columns=["track","points","geometry"])
gdf.plot()


base = NBrabant.plot(color='white', edgecolor="black")
gdf.plot(ax=base)