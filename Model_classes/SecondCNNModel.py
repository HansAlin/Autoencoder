from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D, Activation, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, UpSampling1D, LeakyReLU
from keras_flops import get_flops
from tensorflow.keras import backend as K
import numpy as np
import glob
import Help_functions.plot_functions as pf
import Help_functions.data_manage as dm

class SecondCNNModel:
  def build(data, filters=[32,64,128], activation_function='relu', latent_size=6, kernel=3, last_activation_function='linear', convs=2):
    input_data = keras.Input(shape=data[0].shape)
    layers = len(filters)
    layer = input_data
    for f in filters:
      for j in range(convs):
        layer = Conv1D(filters=f, kernel_size=kernel, activation=activation_function, strides=1, padding='same')(layer)
      layer = MaxPooling1D(pool_size=2)(layer)

    volumeSize = K.int_shape(layer)
    layer = Flatten()(layer)
    latent = Dense(latent_size)(layer)  

    encoder = keras.Model(input_data, latent, name='encoder')

    latentInputs = keras.Input(shape=(latent_size,1,))

    layer = Dense(np.prod(volumeSize[1:]))(latent)

    layer = Reshape((volumeSize[1], volumeSize[2]))(layer)
    
    for f in filters[::-1]:
      for j in range(convs):
        layer = Conv1D(filters=f, kernel_size=kernel, activation=activation_function, strides=1, padding='same')(layer)
      layer = UpSampling1D(2)(layer)
      layer_shape = layer.shape
      value = layer_shape[1]
      if value == 48:
        layer = keras.layers.ZeroPadding1D(1)(layer)

    layer = Conv1D(filters=1, kernel_size=kernel, strides=1, padding='same', activation=activation_function)(layer)

    if last_activation_function == 'linear': 
       layer = Dense(units=1)(layer)
    else:   
      layer = Dense(units=1,activation=last_activation_function)(layer)


    outputs = Reshape((100,1))(layer) 
    #decoder = keras.Model(latent, outputs, name='decoder')

    autoencoder = keras.Model(input_data, outputs, name='autoencoder')

    return (encoder, None, autoencoder)
  
  def build2(data, filters=[32,64,128], activation_function='relu', latent_size=6, kernel=3, last_activation_function='linear', convs=2):
    input_data = keras.Input(shape=data[0].shape)
    layer = input_data
    layer = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(layer)
    layer = MaxPooling1D(pool_size=2)(layer)
    layer = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(layer)
    layer = MaxPooling1D(pool_size=2)(layer)
    layer = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(layer)
    layer = MaxPooling1D(pool_size=2)(layer)

    volumeSize = K.int_shape(layer)
    layer = Flatten()(layer)
    layer = Dense(latent_size)(layer)

    layer = Dense(np.prod(volumeSize[1:]))(layer)
    layer = Reshape((volumeSize[1], volumeSize[2]))(layer)

    layer = Conv1D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu')(layer)
    layer = UpSampling1D(2)(layer)
    layer = Conv1D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')(layer)
    layer = UpSampling1D(2)(layer)
    layer = Conv1D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')(layer)
    layer = keras.layers.ZeroPadding1D(1)(layer)
    layer = UpSampling1D(2)(layer)
    outputs = Dense(1)(layer)

    autoencoder = keras.Model(input_data, outputs)
    return (None , None, autoencoder  )
