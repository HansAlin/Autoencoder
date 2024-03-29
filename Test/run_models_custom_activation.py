### Tensorflow 2.2  ###########
import os
from gpuutils import GpuUtils
GpuUtils.allocate(gpu_count=1, framework='keras')
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# ### Tensorflow 2.4  ###########
# import tensorflow as tf
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#   try:
#     # Currently, memory growth needs to be the same across GPUs
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
#     logical_gpus = tf.config.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Memory growth must be set before GPUs have been initialized
#     print(e)

from gc import callbacks
from tensorflow import keras, math
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D, Activation, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, UpSampling1D
from tensorflow.keras import backend as K
from keras_flops import get_flops
import data_manage as dm
import creating_models as cm
import plot_functions as pf
import pandas as pd 
import numpy as np


def activation_function_1(x):
  
  return K.tanh(x)*2**(-3)
def activation_function_2(x):
  
  return K.tanh(x)*2**(-2)
def activation_function_3(x):
  
  return K.tanh(x)*2**(-1)
def activation_function_4(x):
  
  return K.tanh(x)*2**(1) 
def activation_function_5(x):
  
  return K.tanh(x)*2**(2)
def activation_function_6(x):
  
  return K.tanh(x)*2**(3)       

leakyrelu = keras.layers.LeakyReLU()



# Hyper parameters
batches = [1024]
learning_rates = [10**(-4)]
signal_ratios = [0]
kernels = [3]
latent_spaces = [64]#
number_of_filters = [64,128,256]
layers = [1,2,3]
number_of_single_models = 1
single_model = False
epochs = 1000
sub_conv_layers = [1,2]

model_number = 1
test_run = False
all_signals = True
plot =True
small_test_set = 2000
activation_function_bottlenecks= [True]#,True
activation_function_last_layers=['tanh',
                                activation_function_1, 
                                 activation_function_2, 
                                 activation_function_3,
                                 activation_function_4,
                                 activation_function_5,
                                 activation_function_6, 
                                 leakyrelu]#, 

fpr = 0.05
verbose = 1

path = '/home/halin/Autoencoder/Models/CNN_111'
x_test, y_test, smask_test, signal, noise, std, mean = dm.load_data(all_signals=all_signals, small_test_set=small_test_set)


####### Old data and models  ###############3
load_old_results = False
_if_old_model = False
old_model = ''
old_model_path = ''#'/home/halin/Autoencoder/Models/CNN_102/CNN_102_model_5.h5'


if load_old_results:
  results = pd.read_csv(path + '/results.csv')
else:  
  results = pd.DataFrame(columns=[ 'Model name', 'Epochs', 'Batch', 'Kernel', 'Learning rate', 'Sub conv layers', 'Signal ratio', 'False pos.', 'True pos.', 'Threshold value', 'Latent space', 'Number of filters', 'Flops', 'Layers', 'Noise reduction','True pos. array','Signal loss', 'Noise loss', 'Act. last layer','Any act. bottle'])
  results = results.set_index('Model name')
if _if_old_model:
  old_model = load_model(old_model_path)



for batch in batches:
  for learning_rate in learning_rates:
    for signal_ratio in signal_ratios:
      for kernel in kernels:
        for latent_space in latent_spaces:
          for filters in number_of_filters:
            for layer in layers:
              for i in range(number_of_single_models):
                for conv in sub_conv_layers:
                  for activation_function_bottleneck in activation_function_bottlenecks:
                    for activation_function_last_layer in activation_function_last_layers:
                      model_name, total_epochs, batch, kernel, learning_rate, signal_ratio, fpr, tpr, threshold_value, latent_space, filters, flops, layer, noise_reduction_factors, true_pos_array, signal_loss, noise_loss  = cm.create_and_train_model(
                                                              single_model=single_model,
                                                              old_model=old_model,
                                                              layers=layer,
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
                                                              convs=conv,
                                                              epochs=epochs,
                                                              batch=batch,
                                                              activation_function_bottleneck=activation_function_bottleneck,
                                                              activation_function_last_layer=activation_function_last_layer,
                                                              learning_rate=learning_rate,
                                                              signal_ratio=signal_ratio, 
                                                              plot=plot,
                                                              fpr=fpr,
                                                              number_of_filters=filters,
                                                              _old_model = _if_old_model)
                
                      results = results.append({'Model name': model_name,
                                               'Epochs':total_epochs,   
                                               'Batch': batch, 
                                               'Kernel':kernel, 
                                               'Learning rate':learning_rate, 
                                               'Sub conv layers': conv, 
                                               'Signal ratio':signal_ratio, 
                                               'False pos.':fpr, 
                                               'True pos.':tpr, 
                                               'Threshold value':threshold_value, 
                                               'Latent space':latent_space, 
                                               'Number of filters':filters, 
                                               'Flops':flops, 'Layers':layer, 
                                               'Noise reduction':noise_reduction_factors,
                                               'True pos. array':true_pos_array,
                                               'Signal loss':signal_loss,
                                               'Noise loss':noise_loss,
                                               'Act. last layer':activation_function_last_layer,
                                               'Any act. bottle':activation_function_bottleneck},
                                               ignore_index=True)
                
                      model_number += 1


results.to_csv(path + '/results.csv')

pf.plot_table(path)
# best_model_path = '/home/halin/Autoencoder/Models/Best_models/best_model.csv'
# best_model = pd.read_csv(best_model_path)

pf.noise_reduction_from_results(pd.read_csv(path + '/results.csv'), x_low_lim=0.8, save_path= path, name_prefix='', best_model='' )
