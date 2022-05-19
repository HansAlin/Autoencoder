import os
from gpuutils import GpuUtils
GpuUtils.allocate(gpu_count=1, framework='keras')

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

from matplotlib import pyplot as plt
import numpy as np
#import seaborn as sns
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D, Activation, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, UpSampling1D
from keras_flops import get_flops 
from plot_functions import noise_reduction_curve_multi_models
from creating_models import load_models

path = '/home/halin/Autoencoder/Models/CNN_002'
models = load_models(path)
_ = noise_reduction_curve_multi_models(models, path, fpr=0.05, save_outputs=False )