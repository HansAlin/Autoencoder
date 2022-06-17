
from gpuutils import GpuUtils

GpuUtils.allocate(gpu_count=1, framework='keras')

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

from contextlib import redirect_stdout
import os

import numpy as np
import seaborn as sns
import pandas as pd

from scipy import integrate
from matplotlib import pyplot as plt
from tensorflow import keras
from keras_flops import get_flops
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv1D,
                                     Conv2D, Dense, Dropout, Flatten,
                                     GlobalAveragePooling1D,
                                     GlobalAveragePooling2D, MaxPooling1D,
                                     MaxPooling2D, Reshape, UpSampling1D)
from tensorflow.keras.models import Sequential, load_model

import sys

import Help_functions.creating_models as cm
import Help_functions.plot_functions as pf
import Help_functions.data_manage as dm
from Model_classes.NewPhysicsAutoencoder import NewPhysicsAutoencoder
from Model_classes.SecondCNNModel import SecondCNNModel
from Model_classes.DenseModel import DenseModel
from Model_classes.ConvAutoencoder import ConvAutoencoder

# x_test = np.random.rand(1000,100,1)
# (encoder, decoder, autoencoder) = ConvAutoencoder.build(data=x_test,
# 																												filters=[32,64,128], 
# 																												activation_function='relu',
# 																												latent_size=64,
# 																												kernel=3,
# 																												last_activation_function='linear' )
		
#################   Test performance  #########################
# data_url = '/home/halin/Autoencoder/Data/'
# x_test, y_test, smask_test, signal, noise, std, mean = dm.load_data(all_signals=False,
# 																																		 data_path=data_url, 
# 																																		 small_test_set=1000,
# 																																		 number_of_files=10)
# folder = 119
# nr_of_models = 12
# save_path = f'/home/halin/Autoencoder/Models/CNN_{folder}/'
# for i in range(1,nr_of_models + 1): 
# 	dec, en  = divmod(i,10)
# 	if dec == 0:
# 		dec = ''
# 	model_path = f'/home/halin/Autoencoder/Models/CNN_{folder}/CNN_{folder}_model_{dec}{en}.h5'

# 	pf.plot_performance(path=model_path, x_test=x_test,smask_test=smask_test,save_path=save_path,std=std,mean=mean)


# ##########    Create a new table   #################3
# folder_path  = '/home/halin/Autoencoder/Models/'
# folder = 124

# pf.plot_table(folder_path + f'CNN_{folder}', headers=['Model name',
# 																'Epochs',
#                                 'Act. last layer',
#                                 'Activation func. rest'],
# 																sufix='_1_') 

################  Create a sub table  ##################
# folder_path  = '/home/halin/Autoencoder/Models/'
# folder = 119

# csv_path = folder_path + f'CNN_{folder}' + '/results.csv'
# save_path = f'/home/halin/Autoencoder/Models/CNN_{folder}'
# results = pd.read_csv(csv_path)
# sub_results = results[1::2]
# sub_results.to_csv(save_path + '/Sub_results.csv')
# pf.noise_reduction_from_results(results=sub_results,
# 																best_model='',
# 																save_path=save_path,
# 																name_prefix='Sub_')

# pf.plot_table(folder_path + f'CNN_{folder}',
# 																table_name='Sub_results.csv',
# 																 headers=['Model name',
# 																'Epochs',
# 																'Number of filters',
# 																'Latent space',
#                                 'Act. last layer',
#                                 'Activation func. rest'],
# 																sufix='_1_') 																

############   Save model summary   ##########################
# number_of_models = 7
# folder = 'CNN_111'
# for i in range(number_of_models):
#     path = '/home/halin/Autoencoder/Models/' + folder +'/'+ folder + f'_model_{i+1}.h5'
#     model = load_model(path)
#     print(model.summary())
#     with open(path[:-2] + '_summary.txt', 'w') as f:
#         with redirect_stdout(f):
#             model.summary()

############  Add models to a dataframe   ##############
# folder = '111'
# folder_path = '/home/halin/Autoencoder/Models/'
# load_path = '/home/halin/Autoencoder/Models/' + 'CNN_' + folder
# data_url = '/home/halin/Autoencoder/Data/'
# save_path = load_path + '/' + folder
# models = cm.load_models(path=load_path)
# latent_space = [1600, 1600,400,800, 1600,96,3200]
# epochs = [700,230,600,300,1000,1000,250]
# filters = [[64,32],[64,32],[64,32,16],[64,32,16],[64,32,16,8],[64,32,16,8],[128,64]]
# conv_in_row = [1,2,1,2,1,2,1]
# x_test, y_test, smask_test, signal, noise, std, mean = dm.load_data(all_signals=(not False),
# 																																		 data_path=data_url, 
# 																																		 small_test_set=1000,
# 																																		 number_of_files=10)

# results = pd.DataFrame(columns=['Model name',
#                                 'Epochs',
#                                 'Batch', 
#                                 'Kernel', 
#                                 'Learning rate', 
#                                 'False pos.', 
#                                 'True pos.', 
#                                 'Threshold value', 
#                                 'Latent space', 
#                                 'Number of filters', 
#                                 'Conv_in_row',
#                                 'Flops',
#                                 'Layers', 
#                                 'Noise reduction',
#                                 'True pos. array',
#                                 'Signal loss',
#                                 'Noise loss',
#                                 'Act. last layer',
#                                 'Activation func. rest'])

# for i, model in enumerate(models):
#   model_name = f'{folder}_model_{i+1}'
#   signal_loss, noise_loss = pf.prep_loss_values(model,x_test,smask_test)
#   bins = pf.hist(save_path, signal_loss, noise_loss, plot=False)
#   threshold_value, tpr, fpr, tnr, fnr, noise_reduction_factors, true_pos_array = pf.noise_reduction_curve_single_model(
# 																																						model_name=model_name,
# 																																						save_path=save_path, 
# 																																						x_test=x_test, 
# 																																						smask_test=smask_test, 
# 																																						fpr=0.05, 
# 																																						plot=False, 
# 																																						signal_loss=signal_loss, 
# 																																						noise_loss=noise_loss)
#   flops = get_flops(model)
#   results = results.append({'Model name': model_name,
# 													'Epochs':epochs[i],   
# 													'Batch': 1024, 
# 													'Kernel':3, 
# 													'Learning rate':0.0001, 
# 													'False pos.':fpr, 
# 													'True pos.':tpr, 
# 													'Threshold value':threshold_value, 
# 													'Latent space':latent_space[i], 
# 													'Number of filters':filters[i], 
#                           'Conv_in_row':conv_in_row[i],
# 													'Flops':flops,
# 													'Layers':len(filters[i]), 
# 													'Noise reduction':noise_reduction_factors,
# 													'True pos. array':true_pos_array,
# 													'Signal loss':signal_loss,
# 													'Noise loss':noise_loss,
# 													'Act. last layer':'relu',
# 													'Activation func. rest':'tanh'},
# 													ignore_index=True)

# results.to_csv(folder_path + f'CNN_{folder}/' + 'results.csv')   
# pf.plot_table(folder_path + f'CNN_{folder}', headers=['Model name',
#                               'Epochs',
#                               'Batch', 
#                               'Kernel', 
#                               'Learning rate', 
#                               'False pos.', 
#                               'True pos.', 
#                               'Threshold value', 
#                               'Latent space',
#                               'Conv_in_row', 
#                               'Number of filters', 
#                               'Flops',
#                               'Layers',
#                               'Act. last layer',
#                               'Activation func. rest'])  
# pf.noise_reduction_from_results(pd.read_csv(folder_path + f'CNN_{folder}' + '/results.csv'), x_low_lim=0.8, save_path= folder_path + f'CNN_{folder}', name_prefix='', best_model='' )
model_path = '/home/halin/Autoencoder/Models/CNN_133/CNN_133_model_1.h5'     
autoencoder = load_model(model_path)
# for i in range(10):
# 	weights = load
