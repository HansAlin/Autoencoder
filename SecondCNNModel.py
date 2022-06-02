from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D, Activation, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, UpSampling1D, LeakyReLU, Conv1DTranspose
from keras_flops import get_flops
from tensorflow.keras import backend as K
import numpy as np
import glob
import plot_functions as pf
import data_manage as dm

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

    latentInputs = keras.Input(shape=(latent_size,))

    latentInputs = keras.Input(shape=(latent_size,))
    layer = Dense(np.prod(volumeSize[1:]))(latentInputs)

    layer = Reshape((volumeSize[1], volumeSize[2]))(layer)
    
    for f in filters[::-1]:
      for j in range(convs):
        layer = Conv1D(filters=f, kernel_size=kernel, activation=activation_function, strides=1, padding='same')(layer)
      layer = UpSampling1D(2)(layer)
      layer_shape = layer.shape
      value = layer_shape[1]
      if value == 48:
        layer = keras.layers.ZeroPadding1D(1)(layer)

    outputs = Conv1D(filters=1, kernel_size=kernel, strides=1, padding='same', activation=last_activation_function)(layer)
    decoder = keras.Model(latentInputs, outputs, name='decoder')

    autoencoder = keras.Model(inputs=input_data, outputs=decoder(encoder(input_data)), name='autoencoder')

    return (encoder, decoder, autoencoder)