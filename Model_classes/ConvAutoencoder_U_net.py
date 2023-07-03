
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D, Activation, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, UpSampling1D, LeakyReLU, Concatenate
from keras_flops import get_flops
from tensorflow.keras import backend as K
import numpy as np
import glob
import Help_functions.plot_functions as pf
import Help_functions.data_manage as dm

class ConvAutoencoder_U_net:
  ###
  
  def build(data, filters=[32,64], activation_function='relu', latent_size=6, kernel=3, last_activation_function='linear', convs=1):
    
    input_data = keras.Input(shape=data[0].shape)

    layer_1 = Conv1D(filters=filters[0], kernel_size=kernel, strides=1, padding='same', activation=activation_function)(input_data)
    layer_2 = Conv1D(filters=filters[0], kernel_size=kernel, strides=1, padding='same', activation=activation_function)(layer_1)
    layer_3 = BatchNormalization()(layer_2)  
    layer_4 = MaxPooling1D(2)(layer_3)

    layer_5 = Conv1D(filters=filters[1], kernel_size=kernel, strides=1, padding='same', activation=activation_function)(layer_4)
    layer_6 = Conv1D(filters=filters[1], kernel_size=kernel, strides=1, padding='same', activation=activation_function)(layer_5)
    layer_7 = BatchNormalization()(layer_6)  
    layer_8 = MaxPooling1D(2)(layer_7)

     
    layer_9 = Flatten()(layer_8)
    encoder_layer = Dense(latent_size)(layer_9)  

    encoder = keras.Model(input_data, encoder_layer, name='encoder')
    
    layer_10 = Reshape((25, 1))(encoder)
    layer_11 = UpSampling1D()(layer_10)
    layer_12 = Concatenate()([layer_11, 4])
    layer_13 = Conv1D(filters=filters[1], kernel_size=kernel, padding='same', activation=activation_function)(layer_12)
    layer_14 = Conv1D(filters=filters[1], kernel_size=kernel, padding='same', activation=activation_function)(layer_13)

    layer_15 = UpSampling1D()(layer_14)
    layer_16 = Concatenate()([layer_15, input_data])
    layer_17 = Conv1D(filters=filters[1], kernel_size=kernel, padding='same', activation=activation_function)(layer_16)
    layer_18 = Conv1D(filters=filters[1], kernel_size=kernel, padding='same', activation=activation_function)(layer_17)
    layer_19 = Flatten()(layer_18)

    last_layer = Dense(units=1,activation=last_activation_function)(layer_19)
 
    decoder = None #keras.Model(latentInputs, outputs, name='decoder')
    encoder = None
    autoencoder = keras.Model(input_data,last_layer , name='autoencoder')#decoder(encoder(input_data))

    return (encoder, decoder, autoencoder)
