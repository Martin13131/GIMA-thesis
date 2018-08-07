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
from keras.layers import Activation, LSTM, LocallyConnected1D, Conv1D, ZeroPadding1D
from keras.optimizers import Adam, SGD, rmsprop
from sklearn.preprocessing import MinMaxScaler
from keras.layers.wrappers import Bidirectional
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
import keras.backend as K

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

batch_size = 128
epochs = 1000
latent_dim = 256
num_samples = 20
depth = 6
##################################### Preprocessing
gdf = pd.read_pickle("Obfuscations.pickle")

predictor = gdf['np_rp'].iloc[:num_samples]
response = gdf['np_track'].iloc[:num_samples]
predictor = pad_sequences(predictor.values, 200)
response = pad_sequences(response.values, 200)

input_data = np.zeros( (num_samples, 200, 2), dtype='float32')
target_data = np.zeros( (num_samples, 200, 2), dtype='float32')
Translation_table = np.zeros((num_samples, 2))

for i, (predictor_val, response_val) in enumerate(zip(predictor, response)):
    nonzero = predictor_val[:,0] != 0 ## Since padding is the only possible reason for 0 this works
    xmean = np.mean(predictor_val[nonzero,0])
    ymean = np.mean(predictor_val[nonzero,1])
    
    Translation_table[i] = xmean, ymean
    for t, point in enumerate(predictor_val):
        if point[0] == 0:
            continue            
        input_data[i, t, 0] = point[0] - xmean
        input_data[i, t, 1] = point[1] - ymean
    for t, point in enumerate(response_val):
        target_data[i, t, 0] = point[0] - xmean
        target_data[i, t, 1] = point[1] - ymean
            
input_shape = predictor.shape[1:]
#pred = np.random.random([500,5,5])
#res = pred + 2 + np.random.random(1)
#input_shape= pred.shape[1:]



model = Sequential()
model.add(GRU(units=latent_dim, input_shape=input_shape, return_sequences=True))
for i in range(depth):
    model.add(GRU(latent_dim, return_sequences=True))
#model.add(GRU(units=200, return_sequences=True))
#model.add(GRU(units=200, return_sequences=True))
#model.add(LSTM(units=5, input_shape=input_shape, return_sequences=True))
#model.add(Dense(5, input_shape=input_shape, activation='relu'))
#model.add(GRU(32, input_shape=input_shape, return_sequences=True))
#model.add(Bidirectional(LSTM(12, return_sequences=True), input_shape=input_shape))
#model.add(GRU(16, return_sequences=True))
model.add(Dense(2))
#model.add(Conv1D(5, 5))
#model.add(ZeroPadding1D(padding=2))
#Seq2Seq(output_dim=output_dim, hidden_dim=hidden_dim, output_length=output_length, input_shape=(input_length, input_dim), peek=True, depth=2, teacher_force=True)


model_json = model.to_json()
with open("GRUDeepModel.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
    
model.save_weights("GRUDeepmodel.h5")
print("Saved model to disk")
#model.add(Dense(2))
model.compile(optimizer=rmsprop(lr=0.05), loss=my_mean_squared_error) # MSE over actual - rp is 72070.6
model.summary()
model.fit(predictor, response, epochs=epochs, shuffle=True, validation_split=0, callbacks=[EarlyStopping('loss',patience=10), TensorBoard('DeepGRU'), ReduceLROnPlateau(monitor='loss', factor=0.5, patience=8, mode='min', min_lr=0.00001, verbose=1)])
#
frequency = 250# Set Frequency To 2500 Hertz
duration = 1000
while True:
  # Set Duration To 1000 ms == 1 second
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
