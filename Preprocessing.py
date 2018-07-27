# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 16:21:40 2018

@author: mljmo
"""
# Import modules
import os
import pandas as pd
import geopandas as gpd
from shapely import geometry

## Read individual track csv files and concatenate into single dataframe
Path = r"C:\Users\mljmo\OneDrive\GIMA\Thesis\Data\joinedtracks"
os.chdir(Path)
Data = []
if os.path.isfile('joinedTracks.csv'):
    df = pd.read_csv("joinedTracks.csv")
else:    
    for file in os.listdir():
        Data.append(pd.read_csv(os.path.join(Path,file)))
        os.remove(file)    
    df = pd.concat(Data)
    df.to_csv("joinedTracks.csv")


## Load Noord-Brabant shapefile and remove 
## all tracks not completely in the NBrabant area
os.chdir("..")
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
    Data.append([track, points, lines])

gdf = gpd.GeoDataFrame(Data, columns=["track","actPoints","actLines"])
gdf = gdf.set_geometry("actPoints")
gdf.crs = {"init": 'epsg:4326'}
gdf.plot()

NBrabant = NBrabant.to_crs({"init": 'epsg:28992'})
gdf = gdf.to_crs({"init": 'epsg:28992'})
## Read Road network from fietsersbond network 
Roads = gpd.read_file(r"FietsersBondnetwerk_NL/links.shp")
Roads = Roads[Roads["provincie"]=="Noord-Brabant"]
Roads.crs = {'init':'epsg:28992'}
Roads = Roads[['id', 'geometry']]
Nodes = gpd.read_file(r"FietsersBondnetwerk_NL/nodes.shp")
Nodes = Nodes[Nodes["provincie"]=="Noord-Brabant"]
Nodes.crs = {'init':'epsg:28992'}
Nodes = Nodes[['id', 'geometry']]


## Saving geodataframes
gdf.drop("actLines", axis=1)
if not os.path.isfile(r"Preprocessed/tracks.shp"):
    gdf.to_file(r"Preprocessed/tracks.shp")
    Roads.to_file(r"Preprocessed/Roads.shp")
    Nodes.to_file(r"Preprocessed/Nodes.shp")
    NBrabant.to_file(r"Preprocessed/NBrabant.shp")