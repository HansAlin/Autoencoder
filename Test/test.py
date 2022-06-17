import os

from gpuutils import GpuUtils

GpuUtils.allocate(gpu_count=1, framework='keras')

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

import numpy as np
import seaborn as sns
import pandas as pd
from scipy import integrate
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv1D,
                                     Conv2D, Dense, Dropout, Flatten,
                                     GlobalAveragePooling1D,
                                     GlobalAveragePooling2D, MaxPooling1D,
                                     MaxPooling2D, Reshape, UpSampling1D)
from tensorflow.keras.models import Sequential, load_model

import Help_functions.creating_models as cm
import Help_functions.plot_functions as pf
import Help_functions.data_manage as dm



x_test, y_test, smask_test, signal, noise, std, mean = dm.load_data(True)
# i = 9
# path = f'/home/halin/Autoencoder/Models/CNN_00{i}'
# results = pd.read_csv(path + '/results.csv')
# noise_reduction_from_results(results=results, x_low_lim=0.8, save_path=path)
#plot_table(path, table_name='results.csv', headers=['Model name', 'Epochs', 'Batch', 'Kernel', 'Learning rate', 'Signal ratio', 'False pos.', 'True pos.', 'Latent space','Number of filters', 'Flops', 'Layers'])
#adding_noisereduction_values_to_result_table(path, path + '/' + 'results.csv', x_test=x_test, smask_test=smask_test)

# model = cm.create_autoencoder_model(data=x_test,
#                                     latent_space=6,
#                                     layers=2,
#                                     convs=2,
#                                     kernel=3,
#                                     number_of_filters=256,
#                                     activation_function_last_layer='tanh',
#                                     activation_function_bottleneck=True    )
# print(model.summary())
path = '/home/halin/Autoencoder/Models/test_models'
test_size = 100
noise = x_test[~smask_test]
signal = x_test[smask_test]
print(noise.shape)
print(signal.shape)
noise = np.abs(noise)
noise_integrand_values = np.zeros(test_size)
signal_integrand_values = np.zeros(test_size)
time_range = np.linspace(0,0.1,100)
for i in range(test_size):
    print(noise[i,:].shape)
    noise_integrand_values[i] = integrate.simps(y=noise[i,:], x=time_range)
    signal_integrand_values[i] = integrate.simps(y=signal[i,:], x=time_range)
plt.hist(noise_integrand_values, bins=10, alpha=0.5)
plt.hist(signal_integrand_values, bins=10, alpha=0.5)
plt.savefig(path + '/integration_histogram.png')