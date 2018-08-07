# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 14:24:00 2018
Adjusted obfuscation to use shapely geometries
@author: mljmo
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import os
import shapely
from shapely import geometry     
import matplotlib.pyplot as plt
import random
import time

def plotMultipoint(multiPoint):
    for point in multiPoint:
        plt.scatter(point.x, point.y)
    plt.show
    
def multiplot(Input):
    if type(Input) == shapely.geometry.multipoint.MultiPoint:
        plotMultipoint(Input)
    elif type(Input) == gpd.geoseries.GeoSeries:
       plotMultipoint(mp for mp in Input.values)
    elif type(Input) == gpd.geodataframe.GeoDataFrame:
        for multiPoint in Input["geometry"].values:
            plotMultipoint(multiPoint)
    else:
        print("Error plotting, unexpected input type")
        
######################## Grid masking
def gridMask(track, gridDist=250):
    def formGrid(track, gridDist=250):
        xMin, yMin, xMax, yMax = track.bounds
        gridX = np.arange(xMin, xMax, gridDist)
        gridY = np.arange(yMin, yMax, gridDist)
        #grid = [geometry.Point(x,y)for x in gridX for y in gridY]
        return gridX, gridY
    
    def findClosest(point, gridX, gridY):
        closeX = np.argmin(abs(point.x - gridX))
        closeY = np.argmin(abs(point.y - gridY))
        return geometry.Point(gridX[closeX], gridY[closeY])
    
        
    def snapToGrid(track, gridX, gridY):
        ObfTrack = geometry.MultiPoint([findClosest(point, gridX, gridY) for point in track])
        return ObfTrack
    
    gridX, gridY = formGrid(track, gridDist)
    ObfTrack = snapToGrid(track, gridX, gridY)
    return ObfTrack

##################### Random Perturbation
def perturbationMask(track, std=0.1):
    xMin, yMin, xMax, yMax = track.bounds
    xstd = std * (xMax-xMin)
    ystd = std * (yMax-yMin)
    ObfTrack = geometry.MultiPoint([(point.x +np.random.normal(0, xstd), point.y+np.random.normal(0, ystd)) for point in track])
    return ObfTrack

##################### Crowding

def crowding(track, Roads, buffersize=500, pointsToAdd=1000):
    def getRoadSegments(track, Roads, buffersize=500):    
        Buffer = track.buffer(buffersize)                                       #shapely.ops.cascaded_union(track['geometry'].apply(lambda x: x.buffer(buffersize)))
        return Roads[Roads["geometry"].intersects(Buffer)]

    FakePoints = [point for road in getRoadSegments(track, Roads, buffersize)['geometry'] for point in list(road.coords)]
    try:
        FakePointsToAdd = geometry.MultiPoint(random.sample(FakePoints, pointsToAdd))
    except ValueError:
        FakePointsToAdd = geometry.MultiPoint(FakePoints)
        #print("Not enough points available. Added "+str(len(FakePoints)))
    return track.union(FakePointsToAdd)


################### actual script
# Loading data
Path = r"C:\Users\Maarten\OneDrive\GIMA\Thesis\Data"
os.chdir(Path)


gdf = gpd.read_file(r"Preprocessed/tracks.shp")
gdf.crs ={"init": 'epsg:4326'}
gdf.to_crs(epsg=28992, inplace=True)
Roads = gpd.read_file(r"Preprocessed/Roads.shp")
Nodes = gpd.read_file(r"Preprocessed/Nodes.shp")
NBrabant = gpd.read_file(r"Preprocessed/NBrabant.shp")

###### applying obfuscation
ts = time.time()
gdf['gridMasked'] = gdf['geometry'].apply(gridMask)
print("Grid masking took ", time.time()-ts, " seconds")
gmtime = time.time()-ts
gdf.drop('geometry',axis=1).set_geometry('gridMasked').to_file(r"Obfuscated/GridMasking.shp")

ts = time.time()
gdf['randPert'] = gdf['geometry'].apply(perturbationMask)
print("Random Perturbation took ", time.time()-ts, " seconds")
rptime = time.time()-ts
gdf.drop('geometry',axis=1).set_geometry('randPert').to_file(r"Obfuscated/RandomPerturbation.shp")

ts=time.time()
start = 10900
end = start+100
while(end<37000):
    newGdf = gdf[["track", "geometry"]].iloc[start:end].copy()
    newGdf.index = newGdf.track
    newGdf['crowded'] = newGdf['geometry'].apply(lambda track: crowding(track, Roads, buffersize=500, pointsToAdd=1000))
    newGdf.drop('geometry',axis=1).set_geometry('crowded').to_file(r"CrowdingChunks/"+str(start)+".shp")
    start += 100
    end +=100
    print("Crowding took ", time.time()-ts, " seconds, now starting: ", start, end)
crowdTime= time.time() - ts


start = 36900
newGdf = gdf[["track", "geometry"]].iloc[start:].copy()
newGdf.index = newGdf.track
newGdf['crowded'] = newGdf['geometry'].apply(lambda track: crowding(track, Roads, buffersize=500, pointsToAdd=1000))
newGdf.drop('geometry',axis=1).set_geometry('crowded').to_file(r"CrowdingChunks/"+str(start)+".shp")


ToRead = [r"CrowdingChunks/"+str(start)+".shp" for start in range(0,37000,100)]
gdfs = pd.concat([gpd.read_file(x) for x in ToRead])
gdfs.to_file(r"Obfuscated/crowdingObfuscation.shp")



#### change to np arrays
def to_np_array(multipoint):
    return np.array([point.coords for point in multipoint.geoms]).reshape(-1,2)

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

gdf['np_track'] = gdf['geometry'].apply(to_np_array)
gdf['np_rp'] = gdf['randpert'].apply(to_np_array)
gdf['np_gm'] = gdf['gridmasking'].apply(to_np_array)
gdf['np_cr'] = gdf['crowding'].apply(to_np_array)
ToSave = gdf[['np_track','np_gm','np_rp', 'np_cr']]
ToSave.to_pickle("Obfuscations.pickle")

