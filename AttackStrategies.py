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
from keras.preprocessing.sequence import pad_sequences
from keras.layers import GRU, Input, Dense, TimeDistributed, Embedding, Bidirectional, RepeatVector, Flatten
from keras.models import Model, Sequential
from keras.layers import Activation, LSTM, LocallyConnected1D, Conv1D, ZeroPadding1D, Masking
from keras.optimizers import Adam, SGD, rmsprop
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
import keras.backend as K
import time
import winsound

def plotNP(toPlot, cols):
    for col in cols:
        for x in toPlot[col].values:
            plt.scatter(x[:,0],x[:,1]) 
    
def my_mean_squared_error(y_true, y_pred):
    mask = K.not_equal(y_true, 0)
    return K.mean(K.square(tf.boolean_mask(y_pred,mask) - tf.boolean_mask(y_true,mask)), axis=-1)

def to_np_array(multipoint):
    return np.array([point.coords for point in multipoint.geoms]).reshape(-1,2)


path = r"C:\Users\mljmo\OneDrive\GIMA\Thesis\Data"
os.chdir(path)

num_samples = 10000
epochs = 100000
batch_size = 128
##################################### Preprocessing
gdf = pd.read_pickle("Obfuscations.pickle")
TempModel = gdf.sample(num_samples, random_state=1)
predictor = TempModel['np_rp']
response = TempModel['np_track']
predictor = pad_sequences(predictor.values, 849, padding='post')
response = pad_sequences(response.values, 849, padding='post')

input_data = np.zeros( (num_samples, 849, 2), dtype='float32')
target_data = np.zeros( (num_samples, 849, 2), dtype='float32')
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
        target_data[i, t, 0] = point[0] - xmean
        target_data[i, t, 1] = point[1] - ymean
            
input_shape = input_data.shape[1:]
try:
    with open("MyLog.txt", 'w') as myFile:
        for depth in range(6):
            for latent_dim in [2, 4, 8, 16, 32, 64, 128, 256, 512]:
                model = Sequential()
                model.add(Masking(mask_value=0., input_shape=input_shape))
                for i in range(depth):
                    model.add(GRU(latent_dim, return_sequences=True))
                model.add(Dense(2))
                
                model.compile(optimizer=rmsprop(lr=0.05), loss="mse") # MSE over actual - rp is 72070.6
                
                name = str(depth)+"deep"+str(latent_dim)+"widemodel"
                myFile.write(name)
                ts = time.time()
                
                history = model.fit(input_data, target_data, epochs=epochs, batch_size=batch_size, validation_split=0.1, callbacks=[EarlyStopping('loss',patience=50), TensorBoard('NN/'+name), ReduceLROnPlateau(monitor='loss', factor=0.5, patience=20, mode='min', min_lr=0.00001, verbose=1)])
                
                myFile.write("Took " + str(time.time() - ts) + " seconds to train")
                myFile.write("Training loss:\n")
                myFile.write(str(history.history["loss"]))
                myFile.write("Test loss:\n")
                myFile.write(str(history.history["val_loss"]))
                
                model_json = model.to_json()
                with open(name+".json", "w") as json_file:
                    json_file.write(model_json)
                # serialize weights to HDF5
                    
                model.save_weights(name+".h5")
                print("Saved model to disk")
except Exception as e:
    print(e)
    frequency = 250# Set Frequency To 2500 Hertz
    duration = 1000# Set Duration To 1000 ms == 1 second
    while True:
        winsound.Beep(frequency, duration)
        time.sleep(1)
            

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
