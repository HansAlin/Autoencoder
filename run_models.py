import os
from gpuutils import GpuUtils

from plot_functions import noise_reduction_curve_multi_models, noise_reduction_from_results, plot_table
GpuUtils.allocate(gpu_count=1, framework='keras')
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

import data_manage as dm
import creating_models as cm
import pandas as pd 
from data_manage import load_data
from creating_models import create_and_train_model, create_autoencoder_model

# Hyper parameters
batches = [1024]
learning_rates = [10**(-4)]
signal_ratios = [0]
kernels = [3]
latent_spaces = [2,8,32,64,128,256]#
number_of_filters = [64]
layers = [1]
epochs = 300

model_number = 1
test_run = False
all_signals = True
plot =True

fpr = 0.05
verbose = 1

x_test, y_test, smask_test, signal, noise, std, mean = load_data(all_signals=all_signals)
results = pd.DataFrame(columns=[ 'Model name', 'Epochs', 'Batch', 'Kernel', 'Learning rate', 'Signal ratio', 'False pos.', 'True pos.', 'Threshold value', 'Latent space', 'Number of filters', 'Flops', 'Layers', 'Noise reduction','True pos. array'])
path = '/home/halin/Autoencoder/Models/CNN_010'



for batch in batches:
  for learning_rate in learning_rates:
    for signal_ratio in signal_ratios:
      for kernel in kernels:
        for latent_space in latent_spaces:
          for filters in number_of_filters:
            for layer in layers:
              results.loc[model_number]= create_and_train_model(layers=layer,
                                                             model_number=model_number,
                                                              latent_space=latent_space,
                                                              test_run=test_run,
                                                              path=path,
                                                              signal=signal,
                                                              noise=noise,
                                                              verbose=verbose,
                                                              x_test=x_test,
                                                              smask_test=smask_test,
                                                              kernel=kernel,
                                                              epochs=epochs,
                                                              batch=batch,
                                                              learning_rate=learning_rate,
                                                              signal_ratio=signal_ratio, 
                                                              plot=plot,
                                                              fpr=fpr,
                                                              number_of_filters=filters)
              model_number += 1


results.to_csv(path + '/results.csv')

plot_table(path)

#noise_reduction_from_results(results=results, x_low_lim=0.8,save_path=path, best_model='')