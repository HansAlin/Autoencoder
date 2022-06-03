
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D, Activation, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, UpSampling1D, LeakyReLU
from keras_flops import get_flops
from tensorflow.keras import backend as K
import numpy as np
import glob
import plot_functions as pf
import data_manage as dm

class NewPhysicsAutoencoder:
  def build(data, filters=[128,128,128,32], activation_function='relu', latent_size=6, kernel=3, last_activation_function='linear'):
    """
      From: Searching for new physics with deep autoencoders
      Arg:
        data: train or test data
        filters: list och 3 filters
    """
       
    input_data = keras.Input(shape=data[0].shape)
    layers = len(filters)
    layer = input_data
    for i, f in enumerate(filters):
      layer = Conv1D(filters=f, kernel_size=kernel, activation=activation_function, padding='same')(layer)
      if i != (layers - 1):
        layer = MaxPooling1D()(layer)
    volumeSize = K.int_shape(layer) 
    layer = Flatten()(layer)
    layer = Dense(filters[-1], activation=activation_function)(layer)
    latent = Dense(latent_size)(layer)

    encoder = keras.Model(input_data, latent, name='encoder')
    latentInputs = keras.Input(shape=(latent_size,1,))

    layer = Dense(filters[-1], activation=activation_function)(latent)
    layer = Dense(filters[0]*100, activation=activation_function)(layer)
    layer = Reshape((volumeSize[1],filters[1]))(layer)
    for i in range(layers - 1):
      f = filters[i]
      layer = Conv1D(filters=f,kernel_size=kernel, activation=activation_function, padding='same')(layer)
      layer = UpSampling1D()(layer)
      layer_shape = layer.shape
      value = layer_shape[1]
      if value == 48:
        layer = keras.layers.ZeroPadding1D(1)(layer)
    layer = Conv1D(1, kernel_size=kernel, padding='same')(layer)
    if last_activation_function == 'linear': 
       layer = Dense(units=100)(layer)
    else:   
      layer = Dense(units=100,activation=last_activation_function)(layer)
    outputs = Reshape((100,1))(layer) 
    

    #decoder = keras.Model(latent, outputs, name='decoder')
    autoencoder = keras.Model(input_data, outputs, name='autoencoder')

    return (encoder, None, autoencoder)