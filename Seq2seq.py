# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 18:25:41 2018

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
from keras.optimizers import Adam, SGD
from sklearn.preprocessing import MinMaxScaler
from keras.layers.wrappers import Bidirectional
import matplotlib.pyplot as plt
import keras.backend as K

def plotNP(toPlot, cols):
    for col in cols:
        for x in toPlot[col].values:
            plt.scatter(x[:,0],x[:,1]) 
    
    
def to_np_array(multipoint):
    return np.array([point.coords for point in multipoint.geoms]).reshape(-1,2)

def my_mean_squared_error(y_true, y_pred):
    mask = tf.not_equal(y_true, 0)
    return K.mean(K.square(tf.boolean_mask(y_pred,mask) - tf.boolean_mask(y_true,mask)), axis=-1)

#def custom_objective(y_true, y_pred):
    
    
batch_size = 128  # Batch size for training.
epochs = 5000  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 20  # Number of samples to train on.

path = r"C:\Users\mljmo\OneDrive\GIMA\Thesis\Data"
os.chdir(path)

gdf = pd.read_pickle("Obfuscations.pickle")
tempModel = gdf.iloc[:num_samples]
predictor = tempModel["np_gm"]
response = tempModel['np_track']
predictor = pad_sequences(predictor.values, 200)
response = pad_sequences(response.values, 200)

#shape is (n_samples, length, xy)
max_encoder_seq_length = predictor.shape[1]
max_decoder_seq_length = response.shape[1]
num_encoder_tokens = 2 # nr of values out
num_decoder_tokens = 2 # nr of values out


encoder_input_data = np.zeros(
    (num_samples, max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_data = np.zeros(
    (num_samples, max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (num_samples, max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

Translation_table = np.zeros((num_samples, num_encoder_tokens))

for i, (predictor_val, response_val) in enumerate(zip(predictor, response)):
    nonzero = predictor_val[:,0] != 0 ## Since padding is the only possible reason for 0 this works
    xmean = np.mean(predictor_val[nonzero,0])
    ymean = np.mean(predictor_val[nonzero,1])
    
    Translation_table[i] = xmean, ymean
    for t, point in enumerate(predictor_val):
        if point[0] == 0:
            continue            
        encoder_input_data[i, t, 0] = point[0] - xmean
        encoder_input_data[i, t, 1] = point[1] - ymean
    for t, point in enumerate(response_val):
        if point[0] == 0:
            continue
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, 0] = point[0] - xmean
        decoder_input_data[i, t, 1] = point[1] - ymean
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, 0] = point[0] - xmean
            decoder_target_data[i, t - 1, 1] = point[1] - ymean
            
            
            
            
            
# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens)
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

############################## GRU
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = GRU(latent_dim, return_state=True)
encoder_outputs, state_h = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h]

decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = GRU(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens)
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
############################## GRU
from keras.optimizers import rmsprop
from keras.callbacks import TensorBoard, EarlyStopping
# Run training
model.compile(optimizer=rmsprop(lr=0.00001), loss=my_mean_squared_error)
model.summary()


model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          #batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2,
          shuffle=True,
          callbacks=[TensorBoard(log_dir="./logs/GRU/128")])
model_json = model.to_json()
with open("GRUmodel.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
    
model.save_weights("GRUmodel.h5")
print("Saved model to disk")



######
#
#
#json_file = open('model.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#loaded_model = keras.models.model_from_json(loaded_model_json)
## load weights into new model
#loaded_model.load_weights("model.h5")
#print("Loaded model from disk")
#model = loaded_model




toTest
for i in range(toTest):
    





#plt.scatter(encoder_input_data[:,:,0], encoder_input_data[:,:,1], c='b')
#decoder_input_data
#plt.scatter(decoder_target_data[:,:,0], decoder_target_data[:,:,1], c='r')













# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


for seq_index in range(100):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)

###### 10 minute seq2seq - keras
#from keras.models import Model
#from keras.layers import Input, LSTM, Dense
## Define an input sequence and process it.
#encoder_inputs = Input(shape=(None, num_encoder_tokens))
#encoder = LSTM(latent_dim, return_state=True)
#encoder_outputs, state_h, state_c = encoder(encoder_inputs)
## We discard `encoder_outputs` and only keep the states.
#encoder_states = [state_h, state_c]
#
## Set up the decoder, using `encoder_states` as initial state.
#decoder_inputs = Input(shape=(None, num_decoder_tokens))
## We set up our decoder to return full output sequences,
## and to return internal states as well. We don't use the 
## return states in the training model, but we will use them in inference.
#decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
#decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
#                                     initial_state=encoder_states)
#decoder_dense = Dense(num_decoder_tokens, activation='softmax')
#decoder_outputs = decoder_dense(decoder_outputs)
#
## Define the model that will turn
## `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
#model = Model([encoder_inputs, decoder_inputs], decoder_outputs)