# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 15:19:09 2018

@author: mljmo
"""


import os
import numpy as np
import pandas as pd
from shapely import geometry
import geopandas as gpd
import networkx as nx

Path = r"C:\Users\mljmo\OneDrive\GIMA\Thesis\Data"
os.chdir(Path)

Roads = gpd.read_file(r"FietsersBondnetwerk_NL/links.shp")
Roads = Roads[Roads["provincie"]=="Noord-Brabant"]
Roads.crs = {'init':'epsg:28992'}
Roads = Roads.to_crs({'init': 'epsg:4326'})

Nodes = gpd.read_file(r"FietsersBondnetwerk_NL/nodes.shp")
Nodes = Nodes[Nodes["provincie"]=="Noord-Brabant"]
Nodes.crs = {'init':'epsg:28992'}
Nodes = Nodes.to_crs({'init': 'epsg:4326'})

NBrabant = gpd.read_file("NBrabant.shp")
NBrabant["Dissolve"] = True
NBrabant = NBrabant.dissolve(by="Dissolve").buffer(500).to_crs({'init': 'epsg:4326'})


