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
    distance=np.mean([track.loc[i].distance(track.loc[i+1]) for i in range(len(track)-1)])
    return track.buffer(distance)
     
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
            return [p1, p2, p3, p4]
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

    xgrid, ygrid = ReconstructGrid(track)
    areas = findPossibilitySpace(track, xgrid, ygrid)
    return areas

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
    IoU = Intersection / y_pred.area.sum()+y_true.area.sum() - Intersection
    return [precision, recall, IoU]

##################################### Preprocessing
gdf = pd.read_pickle("Obfuscations.pickle")
gridMasked = gdf['np_gm']
randomPerturbed = gdf['np_rp']
crowded = gdf['np_cr']
response = gdf['np_track']
del gdf
#Translation_table = np.zeros((len(gdf),2))
#predictor_data = []
#errorTracks = []
#for i, predictor_val in enumerate(predictor):
#    xmean = np.mean(predictor_val[:,0])
#    ymean = np.mean(predictor_val[:,1])    
#    Translation_table[i] = xmean, ymean
#    track = []
#    for t, point in enumerate(predictor_val):
#        track.append([point[0] - xmean, point[1] - ymean])
#    predictor_data.append(track)
    
response_data = [value.tolist() for value in response.values.tolist()]
ValidationPolygons= gpd.GeoSeries([geometry.LineString(track) for track in response_data]).buffer(10)
del response_data
#%% Calculate bounding boxes)
gmboundingBoxes = gpd.GeoSeries([boundingBox(track) for track in gridMasked])
rpboundingBoxes = gpd.GeoSeries([boundingBox(track) for track in randomPerturbed])
crboundingBoxes = gpd.GeoSeries([boundingBox(track) for track in crowded])

gmbbMetrics = calculateMetrics(ValidationPolygons, gmboundingBoxes)
rpbbMetrics = calculateMetrics(ValidationPolygons, rpboundingBoxes)
crbbMetrics = calculateMetrics(ValidationPolygons, crboundingBoxes)

#%% Apply Buffer:
gmBuffer = gpd.GeoSeries([bufferAttack(track) for track in gridMasked])
print("Buffered GM")
rpBuffer = gpd.GeoSeries([bufferAttack(track) for track in randomPerturbed])
print("Buffered RP")
crBuffer = gpd.GeoSeries([bufferAttack(track) for track in crowded])
print("Buffered Cr")

gmBufferMetrics = calculateMetrics(ValidationPolygons, gmBuffer)
rpBufferMetrics = calculateMetrics(ValidationPolygons, rpBuffer)
crBufferMetrics = calculateMetrics(ValidationPolygons, crBuffer)
#%% Smoothing
gmSmoothing = gpd.GeoSeries([Smoothing(track) for track in gridMasked]) # 80 tracks too short
rpSmoothing = gpd.GeoSeries([Smoothing(track) for track in randomPerturbed])
crSmoothing = gpd.GeoSeries([Smoothing(track) for track in crowded])

gmSmoothMetrics = calculateMetrics(ValidationPolygons, gmSmoothing, convPolygons=True)
rpSmoothMetrics = calculateMetrics(ValidationPolygons, rpSmoothing, convPolygons=True)
crSmoothMetrics = calculateMetrics(ValidationPolygons, crSmoothing, convPolygons=True)

#recall = [Recall(response_data[i], SmoothResults[i]+Translation_table[i]) for i in range(len(SmoothResults))]
#precision = [Precision(response_data[i], SmoothResults[i]+Translation_table[i]) for i in range(len(SmoothResults))]
#print("avg recall:", np.mean(recall), "avg precision", np.mean(precision))

#recall2 = [Recall(response_data[i], predictor_data[i] + Translation_table[i]) for i in range(len(response_data))]
#precision2 = [Precision(response_data[i], predictor_data[i] + Translation_table[i]) for i in range(len(response_data))]
#print("avg recall:", np.mean(recall2), "avg precision", np.mean(precision2))

DeGridMaskResults = [deGridMask(track) for track in gridMasked]

with open("FinalModels/HeuristicResults.txt", 'w') as File:
    File.write("Bounding box results:\n")
    File.write(gmbbMetrics)
    File.write(rpbbMetrics)
    File.write(crbbMetrics)
    
############################### Testing
#x = predictor_data[0]
#x = Smoothing(x) + Translation_table[0]
#plt.scatter(x[:,0], x[:,1], c='r')
#y = np.array(response_data[0])
#plt.scatter(y[:,0], y[:,1], c='b')
#z = np.array(predictor_data[0]) + Translation_table[0]
#plt.scatter(z[:,0], z[:,1], c='y') 
