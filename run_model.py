
import matplotlib.pyplot as plt
import pandas as pd

from creating_models import load_models
import creating_models as cm
import plot_functions as pf
import data_manage as dm
from ConvAutoencoder import ConvAutoencoder
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow import keras
from keras_flops import get_flops
from keras.utils.vis_utils import plot_model

x_test, y_test, smask_test, signal, noise, std, mean = dm.load_data(all_signals=False, data_path='C:/Users/hansa/Skola/Project neutrino autoencoder/Code/Data/', small_test_set=1000)
filters = [32,64,128]
layers = len(filters)
activation_function = 'relu'
latent_size = 6
kernel = 3
last_activation_function = 'linear'
learning_rate = 0.0001
epochs = 1
test_run = True
plot=True
batch=1024
verbose=1
fpr=0.05
model_name = 'CNN_203_model_1'
save_path = 'C:/Users/hansa/Skola/Project neutrino autoencoder/Code/Results/' + model_name[:7] +'/'+ model_name
(encoder, decoder, autoencoder) = ConvAutoencoder.build(data=x_test,
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
plot_model(autoencoder, to_file=save_path + '_model_plot.png', expand_nested=True)    
x_train, smask_train, y_train = dm.create_data(signal=signal, noise=noise, test_run=test_run)      
autencoder = cm.train_autoencoder(model=autoencoder, x_train=x_train, epochs=epochs, batch=batch, verbose=verbose)
flops = get_flops(autoencoder)
signal_loss, noise_loss = pf.prep_loss_values(autoencoder,x_test,smask_test)
bins = pf.hist(save_path, signal_loss, noise_loss, plot=plot)
threshold_value, tpr, fpr, tnr, fnr, noise_reduction_factors, true_pos_array = pf.noise_reduction_curve_single_model(model_name=model_name, save_path=save_path, x_test=x_test, smask_test=smask_test, fpr=fpr, plot=plot, signal_loss=signal_loss, noise_loss=noise_loss)

results = pd.DataFrame(columns=['Model name',
                                'Epochs',
                                'Batch', 
                                'Kernel', 
                                'Learning rate', 
                                'False pos.', 
                                'True pos.', 
                                'Threshold value', 
                                'Latent space', 
                                'Number of filters', 
                                'Flops',
                                'Layers', 
                                'Noise reduction',
                                'True pos. array',
                                'Signal loss',
                                'Noise loss',
                                'Act. last layer',
                                'Activation func. rest'])

results = results.append({'Model name': model_name,
            'Epochs':epochs,   
            'Batch': batch, 
            'Kernel':kernel, 
            'Learning rate':learning_rate, 
            'False pos.':fpr, 
            'True pos.':tpr, 
            'Threshold value':threshold_value, 
            'Latent space':latent_size, 
            'Number of filters':filters, 
            'Flops':flops,
            'Layers':layers, 
            'Noise reduction':noise_reduction_factors,
            'True pos. array':true_pos_array,
            'Signal loss':signal_loss,
            'Noise loss':noise_loss,
            'Act. last layer':last_activation_function,
            'Activation func. rest':activation_function},
            ignore_index=True)
results = pd.DataFrame(results) 
results.to_csv(save_path + 'results.csv')  
pf.plot_table(save_path + 'results.csv')  
print()                   