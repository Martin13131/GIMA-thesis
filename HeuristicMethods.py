# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 14:15:53 2018

@author: mljmo
"""

import os
import numpy as np
import pandas as pd
#import geopandas as gpd
#from shapely import geometry
import scipy
from scipy import spatial
from scipy import signal
### scipy.stats.gaussian_kde

path = r"C:\Users\mljmo\OneDrive\GIMA\Thesis\Data"
os.chdir(path)
    
#####################################
def Smoothing(track, window=3):
    def movingaverage(values, window):
        return np.convolve(values, np.repeat(1.0, window)/window, 'same')
    track = np.array(track)
    track[:,0] = movingaverage(track[:,0],3)
    track[:,1] = movingaverage(track[:,1],3)
    return track

def deGridMask(track):
    track = np.array(track)
    def ReconstructGrid(track):
        yUnique = np.unique(track[:,1])
        yDist = (yUnique[1:-1] - yUnique[:-2]).min()
        xUnique = np.unique(track[:,0])
        xDist = (xUnique[1:-1] - xUnique[:-2]).min()
        xmin, xmax = track[:,0].min(), track[:,0].max()
        ymin, ymax = track[:,1].min(), track[:,1].max()
        xgrid = np.arange(xmin, xmax, xDist)
        ygrid = np.arange(ymin, ymax, yDist)
        return xgrid, ygrid
    
    def findPossibilitySpace(xgrid, ygrid):
        return spatial.Voronoi([(x,y) for x in xgrid for y in ygrid])

    def AreaEstimation(track, voronoiGrid):
        
        
        
        
    xgrid, ygrid = ReconstructGrid(track)
    voronoiGrid = findPossibilitySpace(xgrid, ygrid)
    return voronoiGrid
##################################### Preprocessing
gdf = pd.read_pickle("Obfuscations.pickle")

predictor = gdf['np_gm']
response = gdf['np_track']

Translation_table = np.zeros((len(gdf),2))
predictor_data = []

for i, predictor_val in enumerate(predictor):
    xmean = np.mean(predictor_val[:,0])
    ymean = np.mean(predictor_val[:,1])    
    Translation_table[i] = xmean, ymean
    track = []
    for t, point in enumerate(predictor_val):
        track.append([point[0] - xmean, point[1] - ymean])
    predictor_data.append(track)

response_data = [value.tolist() for value in response.values.tolist()]


############################## Testing
x = predictor_data[0]
x = Smoothing(x) + Translation_table[0]
plt.scatter(x[:,0], x[:,1], c='r')
y = np.array(response_data[0])
plt.scatter(y[:,0], y[:,1], c='b')
z = np.array(predictor_data[0]) + Translation_table[0]
plt.scatter(z[:,0], z[:,1], c='y') 
