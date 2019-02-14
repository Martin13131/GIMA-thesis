# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 13:45:43 2018

@author: mljmo
"""

import pandas as pd
#import geopandas as gpd
import os
#from shapely import geometry 
import numpy as np
import keras
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import GRU, Input, Dense, TimeDistributed, Embedding, Bidirectional, RepeatVector, Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Activation, LSTM, LocallyConnected1D, Conv1D, ZeroPadding1D, Masking, Conv1D
from tensorflow.keras.optimizers import Adam, SGD, rmsprop
from tensorflow.keras.activations import relu
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
import tensorflow.keras.backend as K

import time
import winsound
import psutil
def plotNP(toPlot, cols):
    for col in cols:
        for x in toPlot[col].values:
            plt.scatter(x[:,0],x[:,1]) 
    
def my_root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def to_np_array(multipoint):
    return np.array([point.coords for point in multipoint.geoms]).reshape(-1,2)


path = r"C:\Users\mljmo\OneDrive\GIMA\Thesis\Data"
os.chdir(path)

num_samples = 10000
epochs = 1000
batch_size = 128
##################################### Preprocessing
gdf = pd.read_pickle("Obfuscations.pickle")
TempModel = gdf.sample(num_samples, random_state=1)
predictor = TempModel['np_cr']
response = TempModel['np_track']
predictor = pad_sequences(predictor.values, 1763, padding='post')
response = pad_sequences(response.values, 1763, padding='post')

input_data = np.zeros( (num_samples, 1763, 2), dtype='float32')
target_data = np.zeros( (num_samples, 1763, 2), dtype='float32')
Translation_table = np.zeros((num_samples, 2))

for i, (predictor_val, response_val) in enumerate(zip(predictor, response)):
    nonzero = predictor_val[:,0] != 0 ## Since padding is the only possible reason for 0 this works
    xmean = np.mean(predictor_val[nonzero,0])
    ymean = np.mean(predictor_val[nonzero,1])
    
#    xdif = np.max(predictor_val[nonzero,0]) - np.min(predictor_val[nonzero,0])
#    ydif = np.max(predictor_val[nonzero,1]) - np.min(predictor_val[nonzero,1])
#    Translation_table[i,0,:] = xmean, ymean
#    Translation_table[i,1,:] = xdif, ydif
    for t, point in enumerate(predictor_val):
        if point[0] == 0:
            continue            
        input_data[i, t, 0] = point[0] - xmean
        input_data[i, t, 1] = point[1] - ymean
    for t, point in enumerate(response_val):
        if point[0] == 0:
            continue 
        target_data[i, t, 0] = point[0] - xmean
        target_data[i, t, 1] = point[1] - ymean
        
input_shape = input_data.shape[1:]       
# Get reference loss:
from sklearn.metrics import mean_squared_error
print(mean_squared_error(target_data.reshape(-1,2),input_data.reshape(-1,2)))
# Start models
batch_size = 32
Dims = 64
depths = 5
filters = 5
#options = zip(Dims, depths, filters)
Results = []
latent_dim, depth, filterSize = Dims, depths, filters
print(depth, latent_dim, filterSize)
def identity(x): return x
with tf.device('/gpu:1'):
    model = Sequential()
    model.add(keras.layers.Masking( input_shape=input_shape))
    model.add(keras.layers.Lambda(identity))
    model.add(Conv1D(latent_dim, filterSize, padding="same", ))
    for i in range(depth):
        model.add(Conv1D(latent_dim, filterSize, padding="same"))
        model.add(Dense(latent_dim, activation="elu"))
    model.add(Dense(2))
    
    model.compile(optimizer=Adam(lr=0.01), loss=my_root_mean_squared_error) # MSE over actual - rp is 72070.6
    
    x=model.fit(input_data, target_data, epochs=epochs, batch_size=batch_size, validation_split=0.1,\
              callbacks=[ReduceLROnPlateau(monitor='loss', factor=0.9, patience=3, mode='min', min_lr=0.00001, verbose=1),\
                         EarlyStopping(patience=50)])
    model.save(r"FinalModels/CR_Conv_64_5_5_2.hdf5")
    
with tf.device('/gpu:1'):
    model = Sequential()
    model.add(keras.layers.Masking( input_shape=input_shape))
    model.add(keras.layers.Lambda(identity))
    model.add(GRU(latent_dim, return_sequences=True))
    for i in range(depth):
        model.add(GRU(latent_dim, return_sequences=True))
        model.add(Dense(latent_dim, activation="elu"))
    model.add(Dense(2))
    
    model.compile(optimizer=Adam(lr=0.01), loss=my_root_mean_squared_error) # MSE over actual - rp is 72070.6
    
    x=model.fit(input_data, target_data, epochs=epochs, batch_size=batch_size, validation_split=0.1,\
              callbacks=[ReduceLROnPlateau(monitor='loss', factor=0.9, patience=3, mode='min', min_lr=0.00001, verbose=1),\
                         EarlyStopping(patience=50)])
    model.save(r"FinalModels/CR_GRU_64_5_5_2.hdf5")
#testPredictor = pad_sequences(testModel['np_gm'],849)
#actualResult = pad_sequences(testModel['np_track'],849)
#testResult = model.predict(testPredictor)
#
#toPlot = 5
#for i in range(toPlot):
#    nonZero = predictor[i,:,0] != 0
#    plt.scatter(predictor[i,nonZero,0], predictor[i, nonZero, 1], c='b')
#    plt.scatter(testResult[i, nonZero, 0], testResult[i, nonZero, 1], c='r')
#test = gdf.iloc[36000:]
#testpredictor = pad_sequences(test['np_track'],849, padding='post')
#model.predict(testpredictor)
