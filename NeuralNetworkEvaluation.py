# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 18:51:54 2019

@author: mljmo
"""
import keras, os
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely import geometry
from tensorflow.keras.preprocessing.sequence import pad_sequences
def my_root_mean_squared_error(y_true, y_pred):
    mask = K.not_equal(y_true, 0)
    return K.sqrt(K.mean(K.square(tf.boolean_mask(y_pred,mask) - tf.boolean_mask(y_true,mask)), axis=-1))

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

def restoreTranslation(results, translationtable):
    for i in range(len(results)):
        nonzeros = (results[i, :, 0] != 0) & (results[i,:,1] != 0)
        results[i,nonzeros,0]+=translationtable[i,0]
        results[i,nonzeros,1]+=translationtable[i,1]
    return results
#        
def convertToGeometry(results):
    returnValues = gpd.GeoSeries([geometry.LineString([geometry.Point(point) for point in track if (point[0] != 0) & (point[1] != 0)]) for track in results])
    return returnValues.buffer(10)
    
path = r"C:\Users\mljmo\OneDrive\GIMA\Thesis\Data"
os.chdir(path)

num_samples = 10000
gdf = pd.read_pickle("Obfuscations.pickle")
TempModel = gdf.sample(num_samples, random_state=1)
gdf.drop(TempModel.index, inplace=True)
del TempModel

actualTracks = gdf['np_track']
gridMasked = gdf['np_gm']
randomPerturbed = gdf['np_rp']
#crowded = gdf['np_cr']
del gdf

#%% Preprocessing
response_data = [value.tolist() for value in actualTracks.values.tolist()]
ValidationPolygons= gpd.GeoSeries([geometry.LineString(track) for track in response_data]).buffer(10)
del response_data

gmInput = pad_sequences(gridMasked.values, np.max([len(track) for track in gridMasked]), padding='post')
rpInput = pad_sequences(randomPerturbed.values, np.max([len(track) for track in randomPerturbed]), padding='post')
#crInput = pad_sequences(crowded.values, np.max([len(track) for track in crowded]), padding='post')

gmTranslation_table = np.zeros((len(gridMasked),2))
rpTranslation_table = np.zeros((len(randomPerturbed),2))
#crTranslation_table = np.zeros((len(crowded), 2))


for i, (gmVal, rpVal) in enumerate(zip(gridMasked, randomPerturbed)):#, crowded)):, crVal
    gmnonzero = gmVal[:,0] != 0 
    gmxmean = np.mean(gmVal[gmnonzero,0])
    gmymean = np.mean(gmVal[gmnonzero,1])
    gmTranslation_table[i,:] = gmxmean, gmymean
    
    for t, point in enumerate(gmVal):
        if point[0] == 0:
            continue            
        gmInput[i, t, 0] = point[0] - gmxmean
        gmInput[i, t, 1] = point[1] - gmymean
    
    rpnonzero = rpVal[:,0] != 0
    rpxmean = np.mean(rpVal[rpnonzero,0])
    rpymean = np.mean(rpVal[rpnonzero,1])
    rpTranslation_table[i,:] = rpxmean, rpymean
    
    for t, point in enumerate(rpVal):
        if point[0] == 0:
            continue            
        rpInput[i, t, 0] = point[0] - rpxmean
        rpInput[i, t, 1] = point[1] - rpymean
           
#    crnonzero = crVal[:,0] != 0 
#    crxmean = np.mean(crVal[crnonzero,0])
#    crymean = np.mean(crVal[crnonzero,1])
#    crTranslation_table[i,0,:] = crxmean, crymean
#
#    for t, point in enumerate(crVal):
#        if point[0] == 0:
#            continue            
#        crInput[i, t, 0] = point[0] - crxmean
#        crInput[i, t, 1] = point[1] - crymean 

        


gmConv = tensorflow.keras.models.load_model("FinalModels/GM_Conv_64_5_5.hdf5", custom_objects={"my_root_mean_squared_error": my_root_mean_squared_error})
gmConvResults = gmConv.predict(gmInput)
gmConvGeom = convertToGeometry(restoreTranslation(gmConvResults, gmTranslation_table))
gmConvGeom.to_file("Attacked/gmConv.shp")
gmConvMetrics = calculateMetrics(ValidationPolygons, gmConvGeom)
#gmConvMetrics = calculateMetrics(ValidationPolygons, gmConvResults)
del gmConv


gmGru = keras.models.load_model("FinalModels/GM_GRU_64_5_5.hdf5", custom_objects={"my_root_mean_squared_error": my_root_mean_squared_error})
gmGruResults = gmGru.predict(gmInput)
gmGruGeom = convertToGeometry(restoreTranslation(gmGruResults, gmTranslation_table))
gmGruGeom.to_file("Attacked/gmGru.shp")
gmGruMetrics = calculateMetrics(ValidationPolygons, gmGruGeom)
del gmGru, gmGruResults, gmGruGeom


rpConv = keras.models.load_model("FinalModels/RP_Conv_64_5_5.hdf5", custom_objects={"my_root_mean_squared_error": my_root_mean_squared_error})
rpConvResults = rpConv.predict(rpInput)
rpConvGeom = convertToGeometry(restoreTranslation(rpConvResults, rpTranslation_table))
rpConvGeom.to_file("Attacked/rpConv.shp")
rpConvMetrics = calculateMetrics(ValidationPolygons, rpConvGeom)
del rpConv, rpConvResults, rpConvGeom

rpGru = keras.models.load_model("FinalModels/RP_GRU_64_5_5.hdf5", custom_objects={"my_root_mean_squared_error": my_root_mean_squared_error})
rpGruResults = rpGru.predict(rpInput)
rpGruGeom = convertToGeometry(restoreTranslation(rpGruResults, rpTranslation_table))
rpGruGeom.to_file("Attacked/rpGru.shp")
rpGruMetrics = calculateMetrics(ValidationPolygons, rpGruGeom)
del rpGru, rpGruResults, rpGruGeom







#
#crConv = keras.models.load_model("FinalModels/CR_Conv_64_5_5.hdf5", custom_objects={"my_root_mean_squared_error": my_root_mean_squared_error})
#crConvResults = crConv.predict(crInput)
#crConvMetrics = calculateMetrics(ValidationPolygons, crConvResults, convPolygons=True)
#del crConv
#
#crGru = keras.models.load_model("FinalModels/CR_GRU_64_5_5.hdf5", custom_objects={"my_root_mean_squared_error": my_root_mean_squared_error})
#crGruResults = crGru.predict(crInput)
#crGruMetrics = calculateMetrics(ValidationPolygons, crGruResults, convPolygons=True)
#del crGru
#



