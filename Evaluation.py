# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 13:19:49 2018

@author: mljmo
"""
import os
import pandas as pd
#import geopandas as gpd
from shapely import geometry
path = r"C:\Users\mljmo\OneDrive\GIMA\Thesis\Data"
os.chdir(path)


gdf = pd.read_pickle("Obfuscations.pickle")


gdf = gpd.GeoDataFrame(gdf)


def Evaluation(y_true, y_pred):
    y_true_pol = geometry.LineString(y_true).buffer(10)
    y_pred_pol = geometry.LineString(y_pred).buffer(10)
    return y_true_pol.difference(y_pred_pol).area / y_true_pol.area