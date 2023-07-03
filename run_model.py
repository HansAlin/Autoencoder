import os
from re import M
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
from Model_classes.ConvAutoencoder_dropout import ConvAutoencoder_dropout
from tensorflow.keras import backend as K



# Specifiy the hyperparamters

filterss = [[2,4,8]] 														# Specify the number of filters in each layer   
model_number = 1 																# Naming the models 											
conv_in_rows = [1] 															# Number of equal layers in a row, 
activation_functions = ['relu'] 								# Activation function in the layers 
latent_sizes = [2]															# Latent space  
kernels = [9] 																	# Kernel/Filter size
last_activation_functions=['linear']						# Last activation function
learning_rates = [0.001]												# Learning rate
batches=[1024]																	# Size of the batches
epoch_distribution =[2000]											# Number of epochs, if one needs to test differnt number of epochs
model_type ='ConvAutoencoder_dropout'						# Choose a model, these can be used
																								# 'ConvAutoencoder' 
																		 						# 'DenseModel' 
																		 						# 'NewPhysicsAutoencoder'
																		 						# 'SecondCNNModel' # 


# Specify if it a test run or not
# A test run just use a fraction of the data
test_run = True																	# True means 1000 data points

# Specify if plots should be saved or not
plot=True																				# Plots are saved
 
# Specify the ratio of signals in training process 
signal_ratios = [0] 														# 0 means no signals in training set
max_ratio = np.max(signal_ratios)
all_signals = False 
if 0.0 in signal_ratios and len(signal_ratios) == 1:
	all_signals = True

# Specify training supervision
verbose=1																				# Check Keras for 

# Specify False positive rate. Not used 
fpr=0.05  

# Specify the lower limit on the plots of the
# noise reduction curve
x_low_lim = 0.75

# The number of the test set
folder = 999 

# Specify the number of data files that should be used
number_of_data_files_to_load = 10 							# Max 10

# Specify where the data is stored
data_url = '/Data/'

# Specify where to save results from models
folder_path = '/Models/'

results = pd.DataFrame(columns=['Model name',
																'Model type',
                                'Epochs',
                                'Batch', 
                                'Kernel', 
                                'Learning rate', 
                                'False pos.', 
                                'True pos.', 
                                'Threshold value', 
                                'Latent space', 
                                'Number of filters', 
																'Conv. in row',
                                'Flops',
                                'Layers', 
                                'Noise reduction',
                                'True pos. array',
                                'Signal loss',
                                'Noise loss',
                                'Act. last layer',
                                'Activation func. rest',
																'Signal ratio'])
# TODO change back to 1

recycling_model = ''
x_test, y_test, smask_test, signal, noise, std, mean = dm.load_data(all_signals_for_testing=all_signals,
																																		 all_samples=(not test_run),						
																																		 data_path=data_url, 
																																		 small_test_set=1000,
																																		 number_of_files=number_of_data_files_to_load)
plot_examples = np.load('/Data/plot_examples.npy')
number_of_same_model = len(epoch_distribution)

for filters in filterss:
	layers = len(filters)
	for conv_in_row in conv_in_rows:
		for activation_function in activation_functions:
			for latent_size in latent_sizes:
				for kernel in kernels:
					for last_activation_function in last_activation_functions:
						for learning_rate in learning_rates:
							for batch in batches:
								total_epochs = 0
								for i in range(number_of_same_model):
									for signal_ratio in signal_ratios:
										model_name = f'CNN_{folder}_model_{model_number}'
										save_path = folder_path + f'CNN_{folder}/' + model_name
										if number_of_same_model == 1 or i == 0:
											if model_type == 'ConvAutoencoder':
														(encoder, decoder, autoencoder) = ConvAutoencoder.build(data=x_test,
																																					filters=filters, 
																																					activation_function=activation_function,
																																					latent_size=latent_size,
																																					kernel=kernel,
																																					last_activation_function=last_activation_function,
																																					convs=conv_in_row )
											elif model_type == 'ConvAutoencoder_dropout':
												(encoder, decoder, autoencoder) = ConvAutoencoder_dropout.build(data=x_test,
																																					filters=filters, 
																																					activation_function=activation_function,
																																					latent_size=latent_size,
																																					kernel=kernel,
																																					last_activation_function=last_activation_function,
																																					convs=conv_in_row )
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
											epochs = epoch_distribution[i]
										else:
											autoencoder = recycling_model
											epochs = epoch_distribution[i]
											
										adam = keras.optimizers.Adam(learning_rate=learning_rate) 
										autoencoder.compile(
													loss = 'mse',
													optimizer = adam,
													metrics = ['mse','mae','mape'] )
										print(autoencoder.summary())  
										with open(save_path + '_summary.txt', 'w') as f:
														with redirect_stdout(f):
																autoencoder.summary() 
										x_train, smask_train, y_train = dm.create_data(signal=signal, 
																																	noise=noise, 
																																	test_run=test_run, 
																																	signal_ratio=signal_ratio,
																																	maximum_ratio=max_ratio)      
										trained_autoencoder = cm.train_autoencoder(model=autoencoder,
																																x_train=x_train,
																																	epochs=epochs,
																																	batch=batch,
																																		verbose=verbose)
										flops = get_flops(autoencoder)
										autoencoder.save((save_path + '.h5'))
										if plot:
											pf.loss_plot(save_path, trained_autoencoder)
											sufix = 1
											to_plot = np.vstack((plot_examples[:,0], plot_examples[:,2]))
											pf.plot_single_performance(autoencoder,to_plot,save_path,std,mean, sufix=sufix)
											plt.cla()
											sufix = 2
											to_plot = np.vstack((plot_examples[:,1], plot_examples[:,3]))
											pf.plot_single_performance(autoencoder,to_plot,save_path,std,mean, sufix=sufix)
											plt.cla()
										signal_loss, noise_loss = pf.costum_loss_values(autoencoder,x_test,smask_test)
										bins = pf.hist(save_path, signal_loss, noise_loss, plot=plot)
										threshold_value, tpr, fpr, tnr, fnr, noise_reduction_factors, true_pos_array = pf.noise_reduction_curve_single_model(
																																									model_name=model_name,
																																									save_path=save_path,
																																									fpr=fpr, 
																																									plot=plot, 
																																									signal_loss=signal_loss, 
																																									noise_loss=noise_loss,
																																									x_low_lim=x_low_lim)
										if number_of_same_model > 1:
											total_epochs += epochs
											epochs = total_epochs
										results = results.append({'Model name': model_name,
																'Model type':model_type,
																'Epochs':epochs,   
																'Batch': batch, 
																'Kernel':kernel, 
																'Learning rate':learning_rate, 
																'False pos.':fpr, 
																'True pos.':tpr, 
																'Threshold value':threshold_value, 
																'Latent space':latent_size, 
																'Number of filters':filters, 
																'Conv. in rows':conv_in_row,
																'Flops':flops,
																'Layers':layers, 
																'Noise reduction':noise_reduction_factors,
																'True pos. array':true_pos_array,
																'Signal loss':signal_loss,
																'Noise loss':noise_loss,
																'Act. last layer':last_activation_function,
																'Activation func. rest':activation_function,
																'Signal ratio':signal_ratio},
																ignore_index=True)
										recycling_model = autoencoder
										model_number += 1

results.to_csv(folder_path + f'CNN_{folder}/' + 'results.csv')  
pf.plot_table(pd.read_csv(folder_path + f'CNN_{folder}'+ '/results.csv'),
																save_path=folder_path + f'CNN_{folder}/',	
																 headers=['Model name',
                                'Model type',
																'Epochs',
																'Signal ratio',
                                'Batch', 
                                'Kernel', 
                                'Learning rate', 
																'Conv. in rows',
                                'Latent space', 
                                'Number of filters', 
                                'Flops',
                                'Layers',
                                'Act. last layer',
                                'Activation func. rest'])  
pf.noise_reduction_from_results(pd.read_csv(folder_path + f'CNN_{folder}' + '/results.csv'), 
														x_low_lim=x_low_lim, 
														save_path= folder_path + f'CNN_{folder}/', 
														name_prefix='', 
														best_model='' )

print()                   