from xml.sax.xmlreader import InputSource
from sklearn.manifold import locally_linear_embedding
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

class ConvAutoencoder:
  def build(data, filters=[32,64,128], activation_function='relu', latent_size=6, kernel=3, last_activation_function='linear'):
    input_data = keras.Input(shape=data[0].shape)

    layer = input_data
   
    for f in filters:
      if activation_function == 'LeakyRelu':
        layer = Conv1D(filters=f, kernel_size=kernel, strides=2, padding='same')(layer)
        layer = LeakyReLU()(layer)
      else:  
        layer = Conv1D(filters=f, kernel_size=kernel, strides=2, padding='same', activation=activation_function)(layer)
      layer = BatchNormalization()(layer)  

    volumeSize = K.int_shape(layer)  
    layer = Flatten()(layer)
    latent = Dense(latent_size)(layer)  

    encoder = keras.Model(input_data, latent, name='encoder')

    latentInputs = keras.Input(shape=(latent_size,))
    layer = Dense(np.prod(volumeSize[1:]))(latentInputs)
    layer = Reshape((volumeSize[1], volumeSize[2]))(layer)
    padding = 'same'
    for f in filters[::-1]:
      _, size, _ = K.int_shape(layer)
      if size == 52:
        layer = keras.layers.Cropping1D(cropping=(1,1))(layer)
      if activation_function == 'LeakyRelu':
        layer = Conv1DTranspose(filters=f, kernel_size=kernel, strides=2, padding=padding)(layer)
        layer = LeakyReLU()(layer)
      else:  
        layer = Conv1DTranspose(filters=f, kernel_size=kernel, strides=2, padding=padding, activation=activation_function)(layer)
      layer = BatchNormalization()(layer) 
  
    layer = Conv1DTranspose(filters=1, kernel_size=kernel, padding='same')(layer)  
    outputs = Activation(last_activation_function)(layer)
      
    decoder = keras.Model(latentInputs, outputs, name='decoder')

    autoencoder = keras.Model(input_data, decoder(encoder(input_data)), name='autoencoder')

    return (encoder, decoder, autoencoder)
