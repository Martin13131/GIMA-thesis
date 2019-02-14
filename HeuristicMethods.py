# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 14:15:53 2018

@author: mljmo
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import geometry
import scipy
from scipy import spatial
from scipy import signal
### scipy.stats.gaussian_kde

path = r"C:\Users\mljmo\OneDrive\GIMA\Thesis\Data"
os.chdir(path)
    
#####################################

def boundingBox(track):
    xmin, ymin, xmax, ymax = gpd.GeoSeries([geometry.Point(point) for point in track]).total_bounds
    bounds = [[xmin, ymin], [xmin, ymax],[xmax,ymax],[xmax,ymin]]
    return geometry.Polygon(bounds)

def bufferAttack(track):
    track = gpd.GeoSeries([geometry.Point(point) for point in track])
    distance = np.mean([track.loc[i].distance(track.loc[i+1]) for i in range(len(track)-1)])
    return track.buffer(distance).unary_union
     
def Smoothing(track, window=3):
    try:
        def movingaverage(values, window):
            return np.convolve(values, np.repeat(1.0, window)/window, 'same')
        track = np.array(track)
        track[:,0] = movingaverage(track[:,0],3)
        track[:,1] = movingaverage(track[:,1],3)
        return track
    except ValueError: # If track is shorter than 3 points
        print("Track too short")
        return track

def deGridMask(track):
    track = np.array(track) # Ensure input format
    
    def ReconstructGrid(track): # Reconstructs original grid assuming regular x and y intervals
        yUnique = np.unique(track[:,1])
        yDist = (yUnique[1:-1] - yUnique[:-2]).min()
        xUnique = np.unique(track[:,0])
        xDist = (xUnique[1:-1] - xUnique[:-2]).min()
        
        xmin, xmax = track[:,0].min()-xDist, track[:,0].max()+xDist+1
        ymin, ymax = track[:,1].min()-yDist, track[:,1].max()+yDist+1
        xgrid = np.arange(xmin, xmax, xDist) # +1 to ensure inclusion of xmax in grid
        ygrid = np.arange(ymin, ymax, yDist) # +1 to ensure inclusion of ymax in grid
        return xgrid, ygrid
    
    def CompletePolygon(area, xgrid, ygrid):
        try:
            xDist = xgrid[1] - xgrid[0]
            yDist = ygrid[1] - ygrid[0]
            p1, p2 = area
            x1, y1 = p1
            x2, y2 = p2
            if x1 == x2 and x1 < xgrid[1]: #
                p3, p4 = np.array([x1-xDist, y1]), np.array([x2-xDist, y2])
            elif x1 == x2 and x1 > xgrid[-2]:
                p3, p4 = np.array([x1+xDist, y1]), np.array([x2+xDist, y2])
            elif y1 == y2 and y1 < ygrid[1]:
                p3, p4 = np.array([x1, y1-yDist]), np.array([x2, y2-yDist])
            elif y1 == y2 and y1 > ygrid[-2]:
                p3, p4 = np.array([x1, y1+yDist]), np.array([x2, y2+yDist])
            return [p1, p2, p4,p3]
        except Exception as e:
            print(area, e)
            
    def findPossibilitySpace(track, xgrid, ygrid):
        def TraverseVoronoi(point, grid, vor): 
            gridId = np.argmin(abs(point-grid))
            regionId = vor.point_region[gridId]
            verticesId = vor.regions[regionId]
            area = [vor.vertices[vertexId] for vertexId in verticesId if vertexId != -1]
            if len(area) != 4:
                area = CompletePolygon(area, xgrid, ygrid)
            return area
        
        grid = [(x,y) for x in xgrid for y in ygrid]
        vor = spatial.Voronoi(grid)
        areas = [TraverseVoronoi(point, grid, vor) for point in track]
        return areas
    try:
        xgrid, ygrid = ReconstructGrid(track)
        areas = gpd.GeoSeries([geometry.Polygon(point) for point in findPossibilitySpace(track, xgrid, ygrid)]).unary_union
        return areas
    except ValueError:
        return gpd.GeoSeries([geometry.Point(point) for point in track]).buffer(10)
#%% Metrics:
    
#def precision(y_true, y_pred):
#    return (y_pred.area - y_pred.difference(y_true).area).sum() / y_pred.area.sum()
#def recall(y_true, y_pred):
#    return (y_true.area - y_true.difference(y_pred).area).sum() / y_true.area.sum()
#def IoU(y_true, y_pred):
#    return y_true.intersection(y_pred).area.sum() / (y_true.difference(y_pred).area + y_true.intersection(y_pred).area).sum()

def polygonConvert(track, Buffer=10):
    return geometry.LineString(track).buffer(Buffer)
    
def calculateMetrics(y_true, y_pred, convPolygons=False, Buffer=10):
    if convPolygons:
        y_pred = gpd.GeoSeries([polygonConvert(track, Buffer) for track in y_pred])
        print("Converted to polygons")
    Intersection = y_true.intersection(y_pred).area.sum()
    print("Calculated intersection", Intersection)
#    Difference = y_true.difference(y_pred).area.sum()
#    print("Calculated difference", Difference)
    precision = Intersection / y_pred.area.sum()
    recall = Intersection / y_true.area.sum()
    IoU = Intersection / (y_pred.area.sum()+y_true.area.sum() - Intersection)
    return [precision, recall, IoU]

##################################### Preprocessing
gdf = pd.read_pickle("Obfuscations.pickle")
gridMasked = gdf['np_gm']
randomPerturbed = gdf['np_rp']
crowded = gdf['np_cr']
response = gdf['np_track']
del gdf

    
response_data = [value.tolist() for value in response.values.tolist()]
ValidationPolygons= gpd.GeoSeries([geometry.LineString(track) for track in response_data]).buffer(10)
del response_data

#%% Baseline
gmBase = gpd.GeoSeries([geometry.LineString(track) for track in gridMasked]).buffer(10)
rpBase = gpd.GeoSeries([geometry.LineString(track) for track in randomPerturbed]).buffer(10)
crBase = gpd.GeoSeries([geometry.LineString(track) for track in crowded]).buffer(10)

gmBaseMetrics = calculateMetrics(ValidationPolygons, gmBase)
rpBaseMetrics = calculateMetrics(ValidationPolygons, rpBase)
crBaseMetrics = calculateMetrics(ValidationPolygons, crBase)
#%% Calculate bounding boxes)
gmboundingBoxes = gpd.GeoSeries([boundingBox(track) for track in gridMasked])
gmboundingBoxes.to_file("Attacked/gmbb.shp")
gmbbMetrics = calculateMetrics(ValidationPolygons, gmboundingBoxes)
del gmboundingBoxes

rpboundingBoxes = gpd.GeoSeries([boundingBox(track) for track in randomPerturbed])
rpboundingBoxes.to_file("Attacked/rpbb.shp")
rpbbMetrics = calculateMetrics(ValidationPolygons, rpboundingBoxes)
del rpboundingBoxes

crboundingBoxes = gpd.GeoSeries([boundingBox(track) for track in crowded])
crboundingBoxes.to_file("Attacked/crbb.shp")
crbbMetrics = calculateMetrics(ValidationPolygons, crboundingBoxes)
del crboundingBoxes




#%% Apply Buffer:
gmBuffer = gpd.GeoSeries([bufferAttack(track) for track in gridMasked])
#gmBuffer.to_file("Attacked/gmbuf.shp")
gmBufferMetrics = calculateMetrics(ValidationPolygons, gmBuffer)
del gmBuffer

rpBuffer = gpd.GeoSeries([bufferAttack(track) for track in randomPerturbed])
#rpBuffer.to_file("Attacked/rpbuf.shp")
rpBufferMetrics = calculateMetrics(ValidationPolygons, rpBuffer)
del rpBuffer

crBuffer = gpd.GeoSeries([bufferAttack(track) for track in crowded])
#crBuffer.to_file("Attacked/crbuf.shp")
crBufferMetrics = calculateMetrics(ValidationPolygons, crBuffer)
del crBuffer
#%% Smoothing
gmSmoothing = gpd.GeoSeries([Smoothing(track) for track in gridMasked]) # 80 tracks too short
#gmSmoothing.to_file("Attacked/gmsmooth.shp")
gmSmoothMetrics = calculateMetrics(ValidationPolygons, gmSmoothing, convPolygons=True)
del gmSmoothing

rpSmoothing = gpd.GeoSeries([Smoothing(track) for track in randomPerturbed])
#rpSmoothing.to_file("Attacked/rpsmooth.shp")
rpSmoothMetrics = calculateMetrics(ValidationPolygons, rpSmoothing, convPolygons=True)
del rpSmoothing


crSmoothing = gpd.GeoSeries([Smoothing(track) for track in crowded])
#crSmoothing.to_file("Attacked/crsmooth.shp")
crSmoothMetrics = calculateMetrics(ValidationPolygons, crSmoothing, convPolygons=True)
del crSmoothing

#%% Degridmasking
#DeGridMaskResults = []
#for track in gridMasked:
#    DeGridMaskResults.append(deGridMask(track))
#DeGridMaskResults = [deGridMask(track) for track in gridMasked]


    

