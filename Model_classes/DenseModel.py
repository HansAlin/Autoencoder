from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D, Activation, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, UpSampling1D, LeakyReLU, Dropout
from keras_flops import get_flops
from tensorflow.keras import backend as K
import numpy as np
import glob
import plot_functions as pf
import data_manage as dm

class DenseModel:
  def build(data, filters=[128,64,32], activation_function='relu', latent_size=6, kernel=3, last_activation_function='linear'):
    """
      From: Searching for new physics with deep autoencoders
      Arg:
        data: train or test data
        filters: equal to units
    """
    
    input_data = keras.Input(shape=data[0].shape)
    layer = Flatten()(input_data)
    
    for f in filters:
      layer = Dense(units=f, activation=activation_function)(layer)
      layer = BatchNormalization()(layer)
      layer = Dropout(rate=0.05)(layer)

    layer = Dense(units=latent_size)(layer)
    encoder = keras.Model(input_data, layer, name='encoder')
    latentInputs = keras.Input(shape=(latent_size))
    for f in filters[::-1]:
      layer = Dense(units=f, activation=activation_function)(layer)
      layer = BatchNormalization()(layer)
      layer = Dropout(rate=0.05)(layer)
    if last_activation_function == 'linear': 
       layer = Dense(units=100)(layer)
    else:   
      layer = Dense(units=100,activation=last_activation_function)(layer)
    outputs = Reshape((100,1))(layer) 

    #decoder = keras.Model(latentInputs, outputs, name='decoder')
    autoencoder = keras.Model(input_data, outputs, name='autoencoder')

    return (encoder, None, autoencoder)

