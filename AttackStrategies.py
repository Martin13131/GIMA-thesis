# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 13:45:43 2018

@author: mljmo
"""

import pandas as pd
import geopandas as gpd
import os
from shapely import geometry 

path = Path = r"C:\Users\mljmo\OneDrive\GIMA\Thesis\Data"
os.chdir(path)

original = gpd.read_file(r"Preprocessed/tracks.shp")
original.crs ={"init": 'epsg:4326'}
original.to_crs(epsg=28992, inplace=True)
gm = gpd.read_file(r"Obfuscated/GridMasking.shp")
rp = gpd.read_file(r"Obfuscated/RandomPerturbation.shp")
cr = gpd.read_file(r"Obfuscated/crowdingObfuscation.shp")

gm.columns = ['track','gridmasking']
rp.columns=['track', 'randpert']
cr.columns = ['track','crowding']
gdf = original.merge(gm).merge(rp).merge(cr)
del gm, rp, cr, original
gdf.index = gdf.track
gdf.drop("track",axis=1, inplace=True)

x = gdf.geometry.apply(lambda x: geometry.LineString(x.geoms))
