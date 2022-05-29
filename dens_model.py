### Tensorflow 2.2  ###########
import os
from gpuutils import GpuUtils
GpuUtils.allocate(gpu_count=1, framework='keras')
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)


from gc import callbacks
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D, Activation, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, UpSampling1D
from keras_flops import get_flops
import data_manage as dm
import creating_models as cm
import plot_functions as pf
import pandas as pd 
import numpy as np

def encoder(input):
  layer1 = Dense(units = 50, activation= 'relu', name='layer1')(input)
  layer2 = BatchNormalization(name='layer2')(layer1)
  layer3 = Dropout(rate=0.05, name='layer3')(layer2)

  layer4 = Dense(units = 25, activation= 'relu', name='layer4')(layer3)
  layer5 = BatchNormalization(name='layer5')(layer4)
  layer6 = Dropout(rate=0.05, name='layer6')(layer5)

  bottleneck = Dense(units = 2, name='bottleneck')(layer6)
  return bottleneck

def decoder(bottleneck):
  layer7 = Dense(units=25, activation='relu', name='layer7')(bottleneck)
  layer8 = Dropout(rate=0.05, name='layer8')(layer7)

  layer9 = Dense(units=50, activation='relu', name='layer9')(layer8)
  layer10 = Dropout(rate=0.05, name='layer10')(layer9)

  layer11 = Dense(units=100, activation='tanh')(layer10)
  return layer11

  