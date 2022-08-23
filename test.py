
from cProfile import label
from cmath import nan
from gpuutils import GpuUtils

GpuUtils.allocate(gpu_count=1, framework='keras')

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

from contextlib import nullcontext, redirect_stdout
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
# folder = 128
# nr_of_models = 18

# for i in range(1,nr_of_models + 1): 
# 	save_path = f'/home/halin/Autoencoder/Models/CNN_{folder}/CNN_{folder}_model_{i}'
# 	dec, en  = divmod(i,10)
# 	if dec == 0:
# 		dec = ''
# 	model_path = f'/home/halin/Autoencoder/Models/CNN_{folder}/CNN_{folder}_model_{dec}{en}.h5'
# 	autoencoder = load_model(model_path)
# 	pf.plot_performance(autoencoder,
# 											x_test=x_test,
# 											smask_test=smask_test,
# 											save_path=save_path,
# 											std=std,
# 											mean=mean)


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



##########  Add models to a dataframe   ##############
# folder = '150'
# name_prefix = 'test_'
# folder_path = '/home/halin/Autoencoder/Models/'
# load_path = '/home/halin/Autoencoder/Models/' + 'CNN_' + folder
# data_url = '/home/halin/Autoencoder/Data/'
# save_path = load_path + '/' + folder
# models = cm.load_models(path=load_path)

# filterss = [[32,64,128,256],[32,64,128]]
# conv_in_rows = [1]
# activation_functions = ['relu'] 
# latent_sizes = [64]#
# learning_rates = [0.0001]
# kernels = [3]
# last_activation_functions=['linear', 'tanh']#, 
# epoch_distribution =[11, 1511]#,10,100] # [10,20,60,180,440] #  [10] # Epochs per run
# signal_ratios = [0]
# number_of_same_model = len(epoch_distribution)
# batches=[1024]
# x_test, y_test, smask_test, signal, noise, std, mean = dm.load_data(all_signals_for_testing=True,
# 																																		 all_samples=True,						
# 																																		 data_path=data_url, 
# 																																		 small_test_set=1000,
# 																																		 number_of_files=10)
# model_type ='ConvAutoencoder'
# results = pd.DataFrame(columns=['Model name',
# 								'Model type',
#                                 'Epochs',
#                                 'Batch', 
#                                 'Kernel', 
#                                 'Learning rate', 
#                                 'False pos.', 
#                                 'True pos.', 
#                                 'Threshold value', 
#                                 'Latent space', 
#                                 'Number of filters', 
# 								'Conv. in row',
#                                 'Flops',
#                                 'Layers', 
#                                 'Noise reduction',
#                                 'True pos. array',
#                                 'Signal loss',
#                                 'Noise loss',
#                                 'Act. last layer',
#                                 'Activation func. rest',
# 								'Signal ratio'])
# j = 0
# for filters in filterss:
# 	layers = len(filters)
# 	for conv_in_row in conv_in_rows:
# 		for activation_function in activation_functions:
# 			for latent_size in latent_sizes:
# 				for kernel in kernels:
# 					for last_activation_function in last_activation_functions:
# 						for learning_rate in learning_rates:
# 							for batch in batches:
# 								total_epochs = 0
# 								for i in range(number_of_same_model):
# 									for signal_ratio in signal_ratios:
# 										model = models[j]
# 										model_name = f'{folder}_model_{j+1}'
# 										j += 1
# 										flops = get_flops(model)
# 										epochs = epoch_distribution[i]
# 										signal_loss, noise_loss = pf.prep_loss_values(model,x_test,smask_test)
# 										threshold_value, tpr, fpr, tnr, fnr, noise_reduction_factors, true_pos_array = pf.noise_reduction_curve_single_model(
# 																																																																								model_name=model_name,
# 																																																																								save_path=save_path,
# 																																																																								fpr=0.05, 
# 																																																																								plot=False, 
# 																																																																								signal_loss=signal_loss, 
# 																																																																								noise_loss=noise_loss)
# 										results = results.append({'Model name': model_name,
# 																'Model type':model_type,
# 																'Epochs':epochs,   
# 																'Batch': batch, 
# 																'Kernel':kernel, 
# 																'Learning rate':learning_rate, 
# 																'False pos.':fpr, 
# 																'True pos.':tpr, 
# 																'Threshold value':threshold_value, 
# 																'Latent space':latent_size, 
# 																'Number of filters':filters, 
# 																'Conv. in rows':conv_in_row,
# 																'Flops':flops,
# 																'Layers':layers, 
# 																'Noise reduction':noise_reduction_factors,
# 																'True pos. array':true_pos_array,
# 																'Signal loss':signal_loss,
# 																'Noise loss':noise_loss,
# 																'Act. last layer':last_activation_function,
# 																'Activation func. rest':activation_function,
# 																'Signal ratio':signal_ratio},
# 																ignore_index=True)
																																															
  


# results.to_csv(folder_path + f'CNN_{folder}/' + name_prefix + 'results.csv')
# pf.plot_table(folder_path + f'CNN_{folder}', table_name=name_prefix + 'results.csv', headers=['Model name',
#                                 'Model type',
# 																'Epochs',
# 																'Signal ratio',
#                                 'Batch', 
#                                 'Kernel', 
#                                 'Learning rate', 
# 																'Conv. in rows',
#                                 'Latent space', 
#                                 'Number of filters', 
#                                 'Flops',
#                                 'Layers',
#                                 'Act. last layer',
#                                 'Activation func. rest'])    

# pf.noise_reduction_from_results(pd.read_csv(folder_path + f'CNN_{folder}' + '/results.csv'), 
# 														x_low_lim=0.8, 
# 														save_path= folder_path + f'CNN_{folder}', 
# 														name_prefix=name_prefix, 
# 														best_model='' )


#####################    Create an encoder   ##################333
# data_url = '/home/halin/Autoencoder/Data/'
# x_test, y_test, smask_test, signal, noise, std, mean = dm.load_data(all_signals=(True),
#                                                                                 data_path=data_url, 
#                                                                                 small_test_set=1000,
#                                                                                 number_of_files=10)
# model_path = '/home/halin/Autoencoder/Models/CNN_134/CNN_134_model_2.h5'     
# autoencoder = load_model(model_path)
# saved_weights_path = '/home/halin/Autoencoder/Models/test_models/autoencoder_weights.h5'
# autoencoder.save_weights(saved_weights_path, overwrite = True)
# (encoder, decoder, autoencoder) = ConvAutoencoder.build(data=x_test,
#                                                         filters=[32,64,128,256], 
#                                                         activation_function='tanh',
#                                                         latent_size=2,
#                                                         kernel=3,
#                                                         last_activation_function='linear' )
# encoder.load_weights(saved_weights_path, skip_mismatch = True, by_name = True) 
# encoder.compile(
#       loss = 'mse',
#       optimizer = 'adam',
#       metrics = ['mse','mae','mape'] 
#   )
# encoder.summary()   
# signal_pred_values = encoder.predict(x_test[smask_test]) 
# noise_pred_values = encoder.predict(x_test[~smask_test]) 
# number = 1000
# x = noise_pred_values[:,12]
# y = noise_pred_values[:,1]
# x2 = signal_pred_values[:,12]
# y2 = signal_pred_values[:,1]
# ax1 = plt.hist(x, bins=100, label='Noise', color='blue', alpha=0.5)
# ax2 = plt.hist(x2, bins=100, label='Signal', color='red', alpha=0.5)

# #plt.scatter([[1,2,3,4,5,6,7]]*number,signal_pred_values[:number], color='green', label='Signals', alpha=0.5)  
# # plt.scatter(x,y, color='red', label='Noise', alpha=0.5) 
# # plt.scatter(x2,y2, color='blue',label='Signal', alpha=0.5)
# plt.legend()
# plt.savefig('/home/halin/Autoencoder/Models/test_models/test_encoder_plot')
# plt.show()
# plt.cla()                                                 
# # for i in range(10):
# # 	weights = load


# ######################   Test new loss computation  ###############
######################## and add to dataframe new data ##############
start_folder = 153
end_folder = 154 
x_test, y_test, smask_test, signal, noise, std, mean = dm.load_data()
folder_path = '/home/halin/Autoencoder/Models/'

for folder in range(start_folder, end_folder):
  result_path = folder_path + f'CNN_{folder}/results.csv'
  prefix = 'test_'  # to not interfere with existing data
  results = pd.read_csv(result_path)
  pf.change_new_results(results=results,
                  x_test=x_test,
                  smask_test=smask_test, 
                  prefix=prefix,
                  folder_path=folder_path, 
                  folder=folder) 
  pf.plot_table(folder_path + f'CNN_{folder}', table_name=prefix + 'results.csv', headers=['Model name',								
                                  'Epochs',
                                  'Batch', 
                                  'Kernel', 
                                  'Learning rate', 
                                  'Latent space', 
                                  'Number of filters', 
                                  'Flops',
                                  'True pos.',
                                  'Layers'])  
  pf.noise_reduction_from_results(pd.read_csv(folder_path + f'CNN_{folder}/' + prefix +  'results.csv'), 
                              x_low_lim=0.95, 
                              save_path= folder_path + f'CNN_{folder}', 
                              name_prefix=prefix, 
                              best_model='' )
# # ####################### Find best model based on ####################
####################### reduction curve          ####################

# pf.find_best_model_in_folder(start_model=138,
# 							end_model=139, #exclusive	
#                             number_of_models=10, 
#                             terms_of_condition='',
#                             value_of_condition='',
#                             prefix='test_',
#                             headers=['Model name', 
#                                     'Epochs', 
#                                     'Number of filters',  
#                                     'Kernel', 
#                                     'Batch',
#                                     'Latent space',
#                                     'Act. last layer', 
#                                     'Activation func. rest']) #Act. last layer linear

#x_test, y_test, smask_test, signal, noise, std, mean = dm.load_data()
std = 0.011491077671030752
mean = 2.6521230839856967e-08
plot_examples = np.load('/home/halin/Autoencoder/Data/plot_examples.npy')
start_folder = 153
end_folder = 154
for folder in range(start_folder, end_folder):

    model_number = 1
    folder_path = '/home/halin/Autoencoder/Models/'
    results_path = folder_path + f'CNN_{folder}/results.csv'
    try:
      results = pd.read_csv(results_path)
    except OSError as e:
      print(f'No file in folder CNN_{folder}')
      continue
    (rows, cons) = results.shape
    for model_number in range(1,rows + 1):
        save_path = f'/home/halin/Autoencoder/Models/CNN_{folder}/CNN_{folder}_model_{model_number}'
        model_path = folder_path + f'CNN_{folder}/CNN_{folder}_model_{model_number}.h5'
        model = load_model(model_path)
        sufix = 1
        to_plot = np.vstack((plot_examples[:,0], plot_examples[:,2]))
        pf.plot_single_performance(model,to_plot,save_path,std,mean, sufix=sufix)
        plt.cla()
        sufix = 2
        to_plot = np.vstack((plot_examples[:,1], plot_examples[:,3]))
        pf.plot_single_performance(model,to_plot,save_path,std,mean, sufix=sufix)
        plt.cla()

