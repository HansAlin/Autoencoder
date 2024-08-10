
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

class ConvAutoencoder_dropout:
  def build(data, filters=[32,64,128], activation_function='relu', latent_size=6, kernel=3, last_activation_function='linear', convs=1):
    input_data = keras.Input(shape=data[0].shape)

    layer = input_data
    # TODO chnage to maxpooling and remove stride
   
    for f in filters:
      for j in range(convs):
        if activation_function == 'LeakyRelu':
          layer = Conv1D(filters=f, kernel_size=kernel, strides=1, padding='same')(layer)
          layer = LeakyReLU()(layer)
        else: 
          layer = Conv1D(filters=f, kernel_size=kernel, strides=1, padding='same', activation=activation_function)(layer)
        layer = BatchNormalization()(layer)  
        layer = Dropout(0.2)(layer)
      layer = MaxPooling1D(2)(layer)

    volumeSize = K.int_shape(layer)  
    layer = Flatten()(layer)
    layer = Dense(latent_size, activation=activation_function)(layer)  

    encoder = keras.Model(input_data, layer, name='encoder')
    

    layer = Dense(np.prod(volumeSize[1:]))(layer)
    layer = Reshape((volumeSize[1], volumeSize[2]))(layer)
    padding = 'same'
    for f in filters[::-1]:
      _, size, _ = K.int_shape(layer)
      if size == 52:
        layer = keras.layers.Cropping1D(cropping=(1,1))(layer)
      elif size == 56:
        layer = keras.layers.Cropping1D(cropping=(3,3))(layer) 
      elif size == 48:
        layer = keras.layers.ZeroPadding1D(1)(layer)  
      layer = UpSampling1D()(layer)
      for j in range(convs):
        if activation_function == 'LeakyRelu':
          layer = Conv1D(filters=f, kernel_size=kernel, padding=padding)(layer)
          layer = LeakyReLU()(layer)
        else: 
          layer = Conv1D(filters=f, kernel_size=kernel, padding=padding, activation=activation_function)(layer)
      layer = BatchNormalization()(layer)
      layer = Dropout(0.2)(layer) 
    # TODO check if this is useful ?
    if last_activation_function == 'LeakyRelu':
        layer = Conv1D(filters=f, kernel_size=kernel, padding=padding)(layer)
        layer = LeakyReLU()(layer)
    else:   
      if last_activation_function == 'linear': 
        layer = Conv1D(filters=1, kernel_size=kernel, padding='same')(layer) 
      else:
         layer = Conv1D(filters=1, kernel_size=kernel, padding='same', activation=last_activation_function)(layer)
    
    outputs = Reshape((100,1))(layer)  
    
    latentInputs = keras.Input(shape=(latent_size,))
    decoder = None #keras.Model(latentInputs, outputs, name='decoder')
    
    autoencoder = keras.Model(input_data,outputs , name='autoencoder')#decoder(encoder(input_data))

    return (encoder, decoder, autoencoder)
