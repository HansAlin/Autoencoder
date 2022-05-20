import os

from gpuutils import GpuUtils

GpuUtils.allocate(gpu_count=1, framework='keras')

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

import numpy as np
#import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv1D,
                                     Conv2D, Dense, Dropout, Flatten,
                                     GlobalAveragePooling1D,
                                     GlobalAveragePooling2D, MaxPooling1D,
                                     MaxPooling2D, Reshape, UpSampling1D)
from tensorflow.keras.models import Sequential, load_model


from data_manage import load_data, adding_noisereduction_values_to_result_table
from plot_functions import noise_reduction_curve_multi_models, noise_reduction_from_results, plot_table



x_test, y_test, smask_test, signal, noise, std, mean = load_data(True)
i = 9
path = f'/home/halin/Autoencoder/Models/CNN_00{i}'
results = pd.read_csv(path + '/results.csv')
noise_reduction_from_results(results=results, x_low_lim=0.8, save_path=path)
#plot_table(path, table_name='results.csv', headers=['Model name', 'Epochs', 'Batch', 'Kernel', 'Learning rate', 'Signal ratio', 'False pos.', 'True pos.', 'Latent space','Number of filters', 'Flops', 'Layers'])
#adding_noisereduction_values_to_result_table(path, path + '/' + 'results.csv', x_test=x_test, smask_test=smask_test)

