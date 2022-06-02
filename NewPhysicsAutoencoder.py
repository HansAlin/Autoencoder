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

class NewPhysicsAutoencoder:
  def build(data, filters=[128,128,32], activation_function='relu', latent_size=6, kernel=3, last_activation_function='linear'):
    """
      From: Searching for new physics with deep autoencoders
      Arg:
        data: train or test data
        filters: list och 3 filters
    """
    
    
    input_data = keras.Input(shape=data[0].shape)
    
    layer = Conv1D(filters[0], kernel_size=kernel, activation=activation_function, padding='same')(input)
    layer = MaxPooling1D()(layer)
    layer = Conv1D(filters[1], kernel_size=kernel, activation=activation_function, padding='same')(layer)
    layer = Flatten()(layer)
    layer = Dense(filters[2], activation=activation_function)(layer)
    latent = Dense(latent_size)(layer)

    encoder = keras.Model(input_data, latent, name='encoder')
    latentInputs = keras.Input(shape=(latent_size,))

    layer = Dense(filters[2], activation=activation_function)(encoder)
    layer = Dense(filters[2]*100, activation=activation_function)(layer)
    layer = Reshape((25,filters[1]))(layer)
    layer = Conv1D(filters[0],kernel_size=kernel, activation=activation_function, padding='same')(layer)
    layer = UpSampling1D()(layer)
    layer = Conv1D(1, kernel_size=kernel, padding='same')(layer)
    layer = Reshape((1,100))(layer)
    outputs = Activation(last_activation_function)(layer)  

    decoder = keras.Model(latentInputs, outputs, name='decoder')
    autoencoder = keras.Model(input_data, decoder(encoder(input_data)), name='autoencoder')

    return (encoder, decoder, autoencoder)