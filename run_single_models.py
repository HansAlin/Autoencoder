# ### Tensorflow 2.2  ###########
# import os
# from gpuutils import GpuUtils
# GpuUtils.allocate(gpu_count=1, framework='keras')
# import tensorflow as tf
# physical_devices = tf.config.list_physical_devices('GPU')
# for device in physical_devices:
#     tf.config.experimental.set_memory_growth(device, True)

### Tensorflow 2.4  ###########
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

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


def encoder(input, kernel=3, latent_space=6, number_of_filters=128, layers=3, convs = 1):
  layer = input
  for i in range(layers):
    for j in range(convs):
      layer = Conv1D(filters=number_of_filters, kernel_size=kernel, activation='relu', strides=1, padding='same')(layer)
    layer = MaxPooling1D(pool_size=2)(layer)

  layer = Flatten()(layer)
  encoder = Dense(latent_space)(layer)

  return encoder

def decoder(input, data_size=100, kernel=3, latent_space=6, number_of_filters=128, layers=2, convs=1):
  first_layer_size = data_size
  for j in range(layers):
    first_layer_size = np.int(np.floor(first_layer_size/2))
   
  layer = input
  layer = Dense((first_layer_size*number_of_filters))(layer)
  layer = Reshape((first_layer_size, number_of_filters))(layer)
  
  for i in range(layers):
    for j in range(convs):
      layer = Conv1D(filters=number_of_filters, kernel_size=kernel, activation='relu', strides=1, padding='same')(layer)
    layer = UpSampling1D(2)(layer)
    layer_shape = layer.shape
    value = layer_shape[1]
    if value == 48:
      layer = keras.layers.ZeroPadding1D(1)(layer)


  layer = Conv1D(filters=1, kernel_size=kernel, strides=1, padding='same')(layer)

  return layer  

def autoencoder(input, data_size, kernel, latent_space, number_of_filters, layers,convs):
  enc = encoder(input, kernel, latent_space, number_of_filters, layers,convs)
  autoencoder = decoder(enc, data_size, kernel, latent_space, number_of_filters, layers,convs )
  return autoencoder

def create_autoencoder_model(data,kernel, latent_space, number_of_filters, layers, convs, learning_rate=0.0005):
  data_size = len(data[0])
  adam = keras.optimizers.Adam(learning_rate=learning_rate)
  input_data = keras.Input(shape=data[1].shape)
  
  model = keras.Model(inputs=input_data, outputs=autoencoder(input_data, data_size, kernel, latent_space, number_of_filters, layers=layers, convs=convs))
  model.compile(
      loss = 'mse',
      optimizer = adam,
      metrics = ['mse','mae','mape']   
  )
  return model

def create_and_train_model(single_model, old_model,  layers, model_number, latent_space, test_run, path, signal, noise, verbose, x_test, smask_test, kernel, convs,epochs=5, batch=256, learning_rate=0.0005, signal_ratio=1, plot=False, fpr=0.05, number_of_filters=128, _old_model=False):
  
  prefix = path[-7:]
  

  x_train, smask_train, y_train = dm.create_data(signal, noise, signal_ratio=signal_ratio, test_run=test_run )

  autoencoder_model = ''
  total_epochs = 0
  if _old_model:
    autoencoder_model = old_model
  else:  
    if model_number == 1 or  not single_model:
      autoencoder_model = create_autoencoder_model(x_train, kernel=kernel, latent_space=latent_space, number_of_filters=number_of_filters, layers=layers, convs=convs, learning_rate=learning_rate )
      total_epochs = epochs
    else:
      previus_model_path = path + '/' +prefix + '_model_' + str(model_number - 1) + '.h5'
      autoencoder_model = load_model(previus_model_path) 
      total_epochs = epochs*model_number 

  autoencoder_model.summary()

  model_name = prefix + '_model_' + str(model_number)
  print(model_name)
  path = path + '/' + model_name
  #keras.utils.plot_model(autoencoder_model, to_file=(path + '.jpg'), show_layer_activations=True, show_dtype=True, show_shapes=True)
  trained_autoencoder = cm.train_autoencoder(autoencoder_model,x_train, epochs, batch, verbose)
  signal_loss, noise_loss = pf.prep_loss_values(autoencoder_model,x_test,smask_test)
  if plot:
    pf.loss_plot(path, trained_autoencoder)
  bins = pf.hist(path, signal_loss, noise_loss, plot=plot)
  threshold_value, tpr, fpr, tnr, fnr, noise_reduction_factors, true_pos_array = pf.noise_reduction_curve_multi_models([autoencoder_model],path, x_test=x_test, smask_test=smask_test, fpr=fpr, plot=plot, signal_loss=signal_loss, noise_loss=noise_loss)

  flops = get_flops(autoencoder_model)
  
  autoencoder_model.save((path + '.h5'))

  return model_name, total_epochs, batch, kernel, learning_rate, signal_ratio, fpr, tpr, threshold_value, latent_space, number_of_filters, flops, layers, noise_reduction_factors, true_pos_array  


# Hyper parameters
batches = [1024]
learning_rates = [10**(-4)]
signal_ratios = [0]
kernels = [3]
latent_spaces = [2]#
number_of_filters = [128]
layers = [1]
number_of_single_models = 1
single_model = False
epochs = 10
sub_conv_layers = [1]

model_number = 1
test_run = False
all_signals = True
plot =True
small_test_set = 20000

fpr = 0.05
verbose = 1

path = '/home/halin/Autoencoder/Models/test_models'
x_test, y_test, smask_test, signal, noise, std, mean = dm.load_data(all_signals=all_signals, small_test_set=small_test_set)


####### Old data and models  ###############3
load_old_results = False
_if_old_model = False
old_model = ''
old_model_path = ''


if load_old_results:
  results = pd.read_csv(path + '/results.csv')
else:  
  results = pd.DataFrame(columns=[ 'Model name', 'Epochs', 'Batch', 'Kernel', 'Learning rate', 'Sub conv layers', 'Signal ratio', 'False pos.', 'True pos.', 'Threshold value', 'Latent space', 'Number of filters', 'Flops', 'Layers', 'Noise reduction','True pos. array','Signal loss', 'Noise loss'])
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
                  model_name, total_epochs, batch, kernel, learning_rate, signal_ratio, fpr, tpr, threshold_value, latent_space, filters, flops, layer, noise_reduction_factors, true_pos_array, signal_loss, noise_loss  = cm.create_and_train_model(
                                                              single_model=single_model,
                                                              old_model='',
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
                                               'Sub conv layers': sub_conv_layers, 
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
                                               'Noise loss':noise_loss},
                                               ignore_index=True)
                
                  model_number += 1


results.to_csv(path + '/results.csv')

pf.plot_table(path)
best_model_path = '/home/halin/Autoencoder/Models/Best_models/best_model.csv'
best_model = pd.read_csv(best_model_path)

pf.noise_reduction_from_results(pd.read_csv(path + '/results.csv'), x_low_lim=0.8, save_path= path, name_prefix='Incl_best_model_', best_model=best_model )
