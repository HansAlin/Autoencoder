from gc import callbacks
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D, Activation, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, UpSampling1D
from keras_flops import get_flops
import numpy as np
import glob
from plot_functions import prep_loss_values, loss_plot, hist, noise_reduction_curve_multi_models
#

def encoder(input, kernel=3, latent_space=6, number_of_filters=128, layers=3):
  layer = input
  for i in range(layers):
    layer = Conv1D(filters=number_of_filters, kernel_size=kernel, activation='relu', strides=1, padding='same')(layer)
    layer = Conv1D(filters=number_of_filters, kernel_size=kernel, activation='relu', strides=1, padding='same')(layer)
    layer = MaxPooling1D(pool_size=2)(layer)

  layer = Flatten()(layer)
  encoder = Dense(latent_space)(layer)

  return encoder

def decoder(input, data_size=100, kernel=3, latent_space=6, number_of_filters=128, layers=2):
  first_layer_size = data_size
  for j in range(layers):
    first_layer_size = np.int(np.floor(first_layer_size/2))
   
  layer = input
  layer = Dense((first_layer_size*number_of_filters))(layer)
  layer = Reshape((first_layer_size, number_of_filters))(layer)
  
  for i in range(layers):
    layer = Conv1D(filters=number_of_filters, kernel_size=kernel, activation='relu', strides=1, padding='same')(layer)
    layer = Conv1D(filters=number_of_filters, kernel_size=kernel, activation='relu', strides=1, padding='same')(layer)
    layer = UpSampling1D(2)(layer)
    layer_shape = layer.shape
    value = layer_shape[1]
    if value == 48:
      layer = keras.layers.ZeroPadding1D(1)(layer)


  layer = Conv1D(filters=number_of_filters, kernel_size=kernel, activation='relu', strides=1, padding='same')(layer)
  layer = Conv1D(filters=1, kernel_size=kernel, activation='tanh', strides=1, padding='same')(layer)

  return layer  

def autoencoder(input, data_size, kernel, latent_space, number_of_filters, layers):
  enc = encoder(input, kernel, latent_space, number_of_filters, layers)
  autoencoder = decoder(enc, data_size, kernel, latent_space, number_of_filters, layers )
  return autoencoder

def create_autoencoder_model(data, kernel, latent_space, number_of_filters, layers, learning_rate=0.0005,):
  data_size = len(data[0])
  adam = keras.optimizers.Adam(learning_rate=learning_rate)
  input_data = keras.Input(shape=data[1].shape, name='first_layer')
  
  model = keras.Model(inputs=input_data, outputs=autoencoder(input_data, data_size, kernel, latent_space, number_of_filters, layers=layers))
  model.compile(
      loss = 'mse',
      optimizer = adam,
      metrics = ['mse','mae','mape']   
  )
  return model                

def  train_autoencoder(model, x_train, epochs=50, batch=16, verbose=0):
  early_stopping = keras.callbacks.EarlyStopping(
                                    monitor="mse",
                                    min_delta=0,
                                    patience=5,
                                    verbose=0,
                                    mode="auto",
                                    baseline=None,
                                    restore_best_weights=True,
                                )

  val_split = 0.2
  autoencoder = model.fit(x = x_train,
                          y = x_train,
                          epochs=epochs,
                          batch_size = batch,
                          verbose=verbose,
                          shuffle = True,
                          validation_split = val_split,
                          callbacks=early_stopping
                          )
  return autoencoder

def create_and_train_model(layers, model_number, latent_space, test_run, path, signal, noise, verbose, x_test, smask_test, kernel, epochs=5, batch=256, learning_rate=0.0005, signal_ratio=1, plot=False, fpr=0.05, number_of_filters=128):
  from data_manage import create_data
  prefix = path[-7:]
  model_name = prefix + '_model_' + str(model_number)
  print(model_name)
  path = path + '/' + model_name

  x_train, smask_train, y_train = create_data(signal, noise, signal_ratio=signal_ratio, test_run=test_run )
  
  autoencoder_model = create_autoencoder_model(x_train, kernel=kernel, latent_space=latent_space, number_of_filters=number_of_filters, layers=layers, learning_rate=learning_rate )
  autoencoder_model.summary()
  #keras.utils.plot_model(autoencoder_model, to_file=(path + '.jpg'), show_layer_activations=True, show_dtype=True, show_shapes=True)
  trained_autoencoder = train_autoencoder(autoencoder_model,x_train, epochs, batch, verbose)
  signal_loss, noise_loss = prep_loss_values(autoencoder_model,x_test,smask_test)
  if plot:
    loss_plot(path, trained_autoencoder)
  bins = hist(path, signal_loss, noise_loss, plot=plot)
  threshold_value, tpr, fpr, tnr, fnr, noise_reduction_factors, true_pos_array = noise_reduction_curve_multi_models([autoencoder_model],path, x_test=x_test, smask_test=smask_test, fpr=fpr, plot=plot)

  flops = get_flops(autoencoder_model)
  
  autoencoder_model.save((path + '.h5'))

  return model_name, epochs, batch, kernel, learning_rate, signal_ratio, fpr, tpr, threshold_value, latent_space, number_of_filters, flops, layers, noise_reduction_factors, true_pos_array  

def load_models(path):
  """
    This function search for keras models in an folder and loads
    it to a list en returns a list of models. Models which contains 
    the substring "best_model" are excluded.
    Args:
      path: were to search for models
  """
  
  import glob
  list_of_models = glob.glob(path + '/*.h5')
  models = []
  
  for i, path in enumerate(list_of_models):
    if 'best_model' in path:
      pass
    else: 
      models.append(load_model(path))

  return models
