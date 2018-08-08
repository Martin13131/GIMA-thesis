# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 13:19:49 2018

@author: mljmo
"""
import os
import pandas as pd
import geopandas as gpd
from shapely import geometry
path = r"C:\Users\mljmo\OneDrive\GIMA\Thesis\Data"
os.chdir(path)


gdf = pd.read_pickle("Obfuscations.pickle")


gdf = gpd.GeoDataFrame(gdf)


def Recall(y_true, y_pred, Buffer=25):
    y_true_pol = gpd.GeoSeries(geometry.LineString(y_true)).buffer(Buffer)
    y_pred_pol = gpd.GeoSeries(geometry.LineString(y_pred)).buffer(Buffer)
    return (y_true_pol.area - y_true_pol.difference(y_pred_pol).area) / y_true_pol.area

def Precision(y_true, y_pred, Buffer=25):
    y_true_pol = gpd.GeoSeries(geometry.LineString(y_true)).buffer(Buffer)
    y_pred_pol = gpd.GeoSeries(geometry.LineString(y_pred)).buffer(Buffer)
    return (y_pred_pol.area - y_pred_pol.difference(y_true_pol).area)/y_pred_pol.area
    
import numpy as np
import matplotlib.pyplot as plt
pred = np.array([[1,1], [4,4], [6,6], [8,8], [10,10], [11,9], [13,6.5]])
points = np.array([[0,0], [2.5,2.5], [5,5], [7.5,7.5], [10,10], [12,8],[15,4]])

plt.figure(0)
plt.scatter(points[:,0],points[:,1], c='b')
plt.title("original track")

plt.figure(1)
plt.scatter(points[:,0],points[:,1], c='b')
plt.scatter(pred[:,0], pred[:,1], c='r')
plt.title("Predicted track")

ActLine = gpd.GeoSeries(geometry.LineString(points))
PredLine = gpd.GeoSeries(geometry.LineString(pred))

base = ActLine.plot(color='blue')
PredLine.plot(ax=base, color='red')
plt.title("Lines")

ActPol = gpd.GeoSeries(ActLine.buffer(1))
PredPol = gpd.GeoSeries(PredLine.buffer(1))

base2 = ActPol.plot(color='blue', alpha=0.3)
PredPol.plot(ax=base2, color='red', alpha=0.3)

base3 = ActPol.plot(color='blue', alpha=0.3)
PredPol.plot(ax=base3, color='red', alpha=0.3)
ActPol.difference(PredPol).plot(ax=base3, color='green', alpha=1)
plt.title("recall: "+str((ActPol.area - ActPol.difference(PredPol).area) / ActPol.area))

base4 = ActPol.plot(color='blue', alpha=0.3)
PredPol.plot(ax=base4, color='red', alpha=0.3)
PredPol.difference(ActPol).plot(ax=base4, color='green', alpha=1)
plt.title("Precision: "+ str((PredPol.area - PredPol.difference(ActPol).area)/PredPol.area))

