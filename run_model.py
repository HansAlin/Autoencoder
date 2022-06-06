import os
from gpuutils import GpuUtils
GpuUtils.allocate(gpu_count=1, framework='keras')
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
import matplotlib.pyplot as plt
import pandas as pd


import Help_functions.creating_models as cm
import Help_functions.plot_functions as pf
import Help_functions.data_manage as dm

import numpy as np 
from tensorflow.keras.models import load_model
from tensorflow import keras
from keras_flops import get_flops
from contextlib import redirect_stdout
from Model_classes.NewPhysicsAutoencoder import NewPhysicsAutoencoder
from Model_classes.SecondCNNModel import SecondCNNModel
from Model_classes.DenseModel import DenseModel
from Model_classes.ConvAutoencoder import ConvAutoencoder

filterss = [[32,64,128], [128,128,128],[64,128,256]] # filter in layers [50,25] 50 means filters (or units if dense layer)
																 # in first layer and 25 filters in second layer
conv_in_row = 2
activation_functions = ['relu']
latent_sizes = [2,32,64,256]
kernels = [3]
last_activation_functions = ['tanh']#'linear'
learning_rates = [0.0001]
epochs = 1
test_run = False
plot=True
batches=[1024]
verbose=1
fpr=0.05
folder = 120
number_of_data_files_to_load = 10 # Max 10
data_url = '/home/halin/Autoencoder/Data/'

folder_path = '/home/halin/Autoencoder/Models/'

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
model_type ='SecondCNNModel' #'ConvAutoencoder' #'NewPhysicsAutoencoder'# 'DenseModel' #  
model_number = 1
x_test, y_test, smask_test, signal, noise, std, mean = dm.load_data(all_signals=(not test_run),
																																		 data_path=data_url, 
																																		 small_test_set=1000,
																																		 number_of_files=number_of_data_files_to_load)

for filters in filterss:
	layers = len(filters)
	for activation_function in activation_functions:
		for latent_size in latent_sizes:
			for kernel in kernels:
				for last_activation_function in last_activation_functions:
					for learning_rate in learning_rates:
						for batch in batches:

							model_name = f'CNN_{folder}_model_{model_number}'
							save_path = folder_path + f'CNN_{folder}/' + model_name
							if model_type == 'ConvAutoencoder':
										(encoder, decoder, autoencoder) = ConvAutoencoder.build(data=x_test,
																																	filters=filters, 
																																	activation_function=activation_function,
																																	latent_size=latent_size,
																																	kernel=kernel,
																																	last_activation_function=last_activation_function )
							elif model_type == 'NewPhysicsAutoencoder':
										(encoder, decoder, autoencoder) = NewPhysicsAutoencoder.build(data=x_test,
																																	filters=filters, 
																																	activation_function=activation_function,
																																	latent_size=latent_size,
																																	kernel=kernel,
																																	last_activation_function=last_activation_function )
							elif model_type == 'SecondCNNModel':
										(encoder, decoder, autoencoder) = SecondCNNModel.build(data=x_test,
																																	filters=filters, 
																																	activation_function=activation_function,
																																	latent_size=latent_size,
																																	kernel=kernel,
																																	last_activation_function=last_activation_function,
																																	convs=conv_in_row )
							elif model_type == 'DenseModel':
										(encoder, decoder, autoencoder) = DenseModel.build(data=x_test,
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
							with open(save_path + '_summary.txt', 'w') as f:
											with redirect_stdout(f):
													autoencoder.summary() 
							x_train, smask_train, y_train = dm.create_data(signal=signal, noise=noise, test_run=test_run)      
							trained_autoencoder = cm.train_autoencoder(model=autoencoder,
																													x_train=x_train,
																														epochs=epochs,
																														batch=batch,
																															verbose=verbose)
							flops = get_flops(autoencoder)
							if plot:
								pf.loss_plot(save_path, trained_autoencoder)
							signal_loss, noise_loss = pf.prep_loss_values(autoencoder,x_test,smask_test)
							bins = pf.hist(save_path, signal_loss, noise_loss, plot=plot)
							threshold_value, tpr, fpr, tnr, fnr, noise_reduction_factors, true_pos_array = pf.noise_reduction_curve_single_model(
																																						model_name=model_name,
																																						save_path=save_path, 
																																						x_test=x_test, 
																																						smask_test=smask_test, 
																																						fpr=fpr, 
																																						plot=plot, 
																																						signal_loss=signal_loss, 
																																						noise_loss=noise_loss)

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
							model_number += 1

results.to_csv(folder_path + f'CNN_{folder}/' + 'results.csv')  
pf.plot_table(folder_path + f'CNN_{folder}', headers=['Model name',
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
                                'Act. last layer',
                                'Activation func. rest'])  
pf.noise_reduction_from_results(pd.read_csv(folder_path + f'CNN_{folder}' + '/results.csv'), x_low_lim=0.8, save_path= folder_path + f'CNN_{folder}', name_prefix='', best_model='' )

print()                   