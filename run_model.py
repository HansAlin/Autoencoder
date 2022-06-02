
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
from contextlib import redirect_stdout
from NewPhysicsAutoencoder import NewPhysicsAutoencoder
from SecondCNNModel import SecondCNNModel
from DenseModel import DenseModel


filterss = [[50, 25]] #filter in layers
conv_in_row = 2
activation_functions = ['relu']
latent_sizes = [2,4,8]
kernels = [3]
last_activation_functions = ['tanh','linear']
learning_rates = [0.0001]
epochs = 2
test_run = False
plot=True
batches=[1024]
verbose=1
fpr=0.05
folder = 118
number_of_data_files_to_load = 4 # Max 10
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
model_type = 'DenseModel' #'ConvAutoencoder' #'NewPhysicsAutoencoder'#'SecondCNNModel' #  
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
print()                   