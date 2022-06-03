
import matplotlib.pyplot as plt
import pandas as pd

from creating_models import load_models
import creating_models as cm
import plot_functions as pf
import data_manage as dm
from ConvAutoencoder import ConvAutoencoder
from NewPhysicsAutoencoder import NewPhysicsAutoencoder
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow import keras
from keras_flops import get_flops
from keras.utils.vis_utils import plot_model

x_test, y_test, smask_test, signal, noise, std, mean = dm.load_data(all_signals=False, data_path='C:/Users/hansa/Skola/Project neutrino autoencoder/Code/Data/', small_test_set=1000)
filters = [128,128,62]
layers = len(filters)
activation_function = 'relu'
latent_size = 6
kernel = 3
last_activation_function = 'linear'
learning_rate = 0.0001
epochs = 2
test_run = True
plot=True
batch=1024
verbose=1
fpr=0.05
model_name = 'CNN_204_model_1'
save_path = 'C:/Users/hansa/Skola/Project neutrino autoencoder/Code/Results/' + model_name[:7] +'/'+ model_name
(encoder, decoder, autoencoder) = NewPhysicsAutoencoder.build(data=x_test,
                                                     filters=filters, 
                                                     activation_function=activation_function,
                                                     latent_size=latent_size,
                                                     kernel=kernel,
                                                     last_activation_function=last_activation_function )
adam = keras.optimizers.Adam(learning_rate=learning_rate) 
autoencoder.compile(
      loss = 'mse',
      optimizer = adam,
      metrics = ['mse','mae','mape'] )
print(autoencoder.summary())  
 