
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
		

###########################                        ####################
###########################   Test performance     ####################
###########################                        ####################
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



###########################                        ####################
###########################   Create a new table   ####################
###########################                        ####################
# folder_path  = '/home/halin/Autoencoder/Models/'
# folder = 124

# pf.plot_table(folder_path + f'CNN_{folder}', headers=['Model name',
# 																'Epochs',
#                                 'Act. last layer',
#                                 'Activation func. rest'],
# 																sufix='_1_') 

###########################                        ####################
###########################   Create a sub table   ####################
###########################                        ####################
# 															

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


###########################                              ####################
###########################  Add models to a dataframe   ##############
################### if the program did not come to an end    ####################
# folder = 176
# name_prefix = ''
# folder_path = '/home/halin/Autoencoder/Models/'
# load_path = '/home/halin/Autoencoder/Models/' + f'CNN_{folder}' 

# models = cm.load_models(path=load_path)

# filterss = [[512,256,128]]
# conv_in_rows = [1]
# activation_functions = ['relu'] 
# latent_sizes = [2,4,8,16]#
# learning_rates = [0.0001]
# kernels = [3]
# last_activation_functions=['linear']#, 
# batches=[1024]
# epochs = 1
# epoch_distribution =[1500]#,10,100] # [10,20,60,180,440] #  [10] # Epochs per run
# number_of_same_model = len(epoch_distribution)
# # test_run = False
# # plot=True

# signal_ratios = [0] 

# data_url = '/home/halin/Autoencoder/Data/'
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
# 						        'Conv. in row',
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
# 										signal_loss, noise_loss = pf.costum_loss_values(model,x_test,smask_test)
# 										threshold_value, tpr, fpr, tnr, fnr, noise_reduction_factors, true_pos_array = pf.noise_reduction_curve_single_model(
# 																																																																								model_name=model_name,
# 																																																																								save_path='',
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
																																															
  

# save_path = folder_path + f'CNN_{folder}/' + name_prefix + 'results.csv'
# results.to_csv(save_path)
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
# 														x_low_lim=0.95, 
# 														save_path= folder_path + f'CNN_{folder}', 
# 														name_prefix=name_prefix, 
# 														best_model='' )


###########################    Create an encoder   ####################
###########################                        ####################
###########################                        ####################
# data_url = '/home/halin/Autoencoder/Data/'
# folder = 179
# model_number = 5
# [filters, latent_size] = pf.find_values_from_model(folder=folder,
#                          model_number=model_number,
#                          values_of_interest=['Number of filters', 'Latent space'])

# x_test, y_test, smask_test, signal, noise, std, mean = dm.load_data(all_signals_for_testing=False,
#                                                                 all_samples=True,						
#                                                                 data_path=data_url, 
#                                                                 small_test_set=1000,
#                                                                 number_of_files=10)
# model_path = f'/home/halin/Autoencoder/Models/CNN_{folder}/CNN_{folder}_model_{model_number}.h5'     
# autoencoder = load_model(model_path)
# saved_weights_path = '/home/halin/Autoencoder/Models/test_models/autoencoder_weights.h5'
# autoencoder.save_weights(saved_weights_path, overwrite = True)
# (encoder, decoder, autoencoder) = ConvAutoencoder.build(data=x_test,
#                                                         filters=filters, 
#                                                         activation_function='relu',
#                                                         latent_size=latent_size,
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
# (data_points, dim) = noise_pred_values.shape
# x_noise = [0]*dim
# x_signal = [0]*dim
# for i in range(0,dim):
#     x_noise[i] = noise_pred_values[:,i]
#     x_signal[i] = signal_pred_values[:,i]
# x_middle_point_noise = [0]*dim
# for i in range(0,dim):
#     x_middle_point_noise[i] = sum(x_noise[i])/len(x_noise[i])
# noise_loss = 0
# signal_loss = 0    
# for i in range(0,dim):
#     noise_loss += (x_noise[i] - x_middle_point_noise[i])**2
#     signal_loss += (x_signal[i] - x_middle_point_noise[i])**2
# noise_loss = noise_loss/len(noise_loss)
# signal_loss = signal_loss/len(signal_loss)
# pf.noise_reduction_curve_single_model(model_name=f'Model_{folder}_model_{model_number}',
#                                     fpr=0.95,
#                                     x_low_lim=0.9,
#                                     save_path='/home/halin/Autoencoder/Models/test_models/Encoder_loss',
#                                     signal_loss=signal_loss,
#                                     noise_loss=noise_loss,
#                                     ) 
# # ax1 = plt.hist(x, bins=100, label='Noise', color='blue', alpha=0.5)
# # ax2 = plt.hist(x2, bins=100, label='Signal', color='red', alpha=0.5)

# #plt.scatter([[1,2,3,4,5,6,7]]*number,signal_pred_values[:number], color='green', label='Signals', alpha=0.5)  
# plt.scatter(x_signal[0],x_signal[1], color='blue',label='Signal', alpha=0.1)
# plt.scatter(x_noise[0],x_noise[1], color='red', label='Noise', alpha=0.5) 
# plt.xlim(-1,1)
# plt.ylim(-1,1)
# plt.grid()
# plt.legend()
# plt.savefig('/home/halin/Autoencoder/Models/test_models/test_encoder_plot')
# plt.show()
# plt.cla()                                                 
# # for i in range(10):
# # 	weights = load


#######################   Test new loss computation    ###############
#######################  and add to dataframe new data ##############
#######################                                #################
# start_folder = 181
# end_folder = 182

# x_test, y_test, smask_test, signal, noise, std, mean = dm.load_data()
# folder_path = '/home/halin/Autoencoder/Models/'

# for folder in range(start_folder, end_folder):
#   result_path = folder_path + f'CNN_{folder}/results.csv'
#   prefix = 'test_'  # to not interfere with existing data
#   try:
#     results = pd.read_csv(result_path)
#   except OSError as e:
#     print(f'No file in folder CNN_{folder}')
#     continue
#   pf.change_new_results(results=results,
#                   x_test=x_test,
#                   smask_test=smask_test, 
#                   prefix=prefix,
#                   folder_path=folder_path, 
#                   folder=folder) 
#   pf.plot_table(folder_path + f'CNN_{folder}', table_name=prefix + 'results.csv', headers=['Model name',								
#                                   'Epochs',
#                                   'Batch', 
#                                   'Kernel', 
#                                   'Learning rate', 
#                                   'Latent space', 
#                                   'Number of filters', 
#                                   'Flops',
#                                   'True pos.',
#                                   'Layers'])  
#   pf.noise_reduction_from_results(pd.read_csv(folder_path + f'CNN_{folder}/' + prefix +  'results.csv'), 
#                               x_low_lim=0.95, 
#                               save_path= folder_path + f'CNN_{folder}', 
#                               name_prefix=prefix, 
#                               best_model='' )

####################### Find best model based on ####################
####################### reduction curve          ####################
# #######################                          ####################
# Filter models with linera activation function in last layer
result_path = pf.find_best_model_in_folder(start_model=172,
							              end_model=173, #exclusive	
                            number_of_models=300, 
                            terms_of_condition ='Act. last layer', #Epochs
                            value_of_condition ='linear', #tanh
                            comparison = 'equal',
                            x_low_lim = 0.99,
                            prefix='test_',
                            headers=['Model name', 
                                    'Epochs', 
                                    'Number of filters',  
                                    #'Kernel', 
                                    #'Batch',
                                    #'Flops',
                                    'Latent space',
                                    #'Act. last layer' 
                                    ]) #'Activation func. rest'Act. last layer linear

# Filter models with greater than 150 epochs                                    
pf.find_best_model_in_folder(terms_of_condition='Epochs',
                              value_of_condition=150,
                              number_of_models=15,
                              comparison='greater',
                              result_path=result_path,
                              x_low_lim=0.99,
                              prefix='test_',
                              headers=['Model name', 
                                      'Epochs', 
                                      'Number of filters',  
                                      #'Kernel', 
                                      #'Batch',
                                      #'Flops',
                                      'Latent space',
                                      #'Act. last layer',
                                      ] )

# # Filter models with a certain number of layers                                      
# pf.find_best_model_in_folder(terms_of_condition='Layers',
#                               value_of_condition=4,
#                               number_of_models=50,
#                               comparison = 'equal',
#                               result_path=result_path,
#                               x_low_lim=0.95,
#                               prefix='test_',
#                               headers=['Model name', 
#                                       'Epochs', 
#                                       'Number of filters',  
#                                       #'Kernel', 
#                                       #'Batch',
#                                       #'Flops',
#                                       'Latent space',
#                                       #'Act. last layer',
#                                       ] )                                      

####################                           ######################
####################     Perfomance plots      ######################
####################                           ######################

#x_test, y_test, smask_test, signal, noise, std, mean = dm.load_data()
# std = 0.011491077671030752
# mean = 2.6521230839856967e-08
# plot_examples = np.load('/home/halin/Autoencoder/Data/plot_examples.npy')
# start_folder = 136
# end_folder = 153
# for folder in range(start_folder, end_folder):

    
#     folder_path = '/home/halin/Autoencoder/Models/'
#     results_path = folder_path + f'CNN_{folder}/results.csv'
#     try:
#       results = pd.read_csv(results_path)
#     except OSError as e:
#       print(f'No file in folder CNN_{folder}')
#       continue
#     (rows, cons) = results.shape
#     for model_number in range(1,rows + 1):
#         save_path = f'/home/halin/Autoencoder/Models/CNN_{folder}/CNN_{folder}_model_{model_number}'
#         model_path = folder_path + f'CNN_{folder}/CNN_{folder}_model_{model_number}.h5'
#         try:
#           model = load_model(model_path) 
#         except OSError as e:
#           print(f'No model {model_number}')
#           continue
#         sufix = 1
#         to_plot = np.vstack((plot_examples[:,0], plot_examples[:,2]))
#         pf.plot_single_performance(model,to_plot,save_path,std,mean, sufix=sufix)
#         plt.cla()
#         sufix = 2
#         to_plot = np.vstack((plot_examples[:,1], plot_examples[:,3]))
#         pf.plot_single_performance(model,to_plot,save_path,std,mean, sufix=sufix)
#         plt.cla()


####################                           ######################
####################     Plot table            ######################
####################                           ######################

# pf.plot_table(path='/home/halin/Autoencoder/Models/CNN_127/', 
#               table_name='test_results.csv', 
#               headers=['Model name', 
#                       'Epochs', 
#                       'Number of filters',  
#                       'Kernel', 
#                       'Batch',
#                       'Latent space',
#                       'Act. last layer', 
#                       'Activation func. rest'])
##########################                    #########################
########################## Integration test   #########################
##########################                    #########################
# x_test, y_test, smask_test, signal, noise, std, mean = dm.load_data(True)
# path = '/home/halin/Autoencoder/Models/test_models'
# test_size = 100
# noise = x_test[~smask_test]
# signal = x_test[smask_test]
# print(noise.shape)
# print(signal.shape)
# noise = np.abs(noise)
# noise_integrand_values = np.zeros(test_size)
# signal_integrand_values = np.zeros(test_size)
# time_range = np.linspace(0,0.1,100)
# for i in range(test_size):
#     print(noise[i,:].shape)
#     noise_integrand_values[i] = integrate.simps(y=noise[i,:], x=time_range)
#     signal_integrand_values[i] = integrate.simps(y=signal[i,:], x=time_range)
# plt.hist(noise_integrand_values, bins=10, alpha=0.5)
# plt.hist(signal_integrand_values, bins=10, alpha=0.5)
# plt.savefig(path + '/integration_histogram.png')

####################                        ###############################
################# PLOT WEIRD SIGNALS AND NOISE ###########################
####################                        ###############################
# model = load_model('/home/halin/Autoencoder/Models/CNN_163/CNN_163_model_10.h5') 
# x_test, y_test, smask_test, signal, noise, std, mean = dm.load_data(True)
# signal_loss, noise_loss = pf.prep_loss_values(model,x_test,smask_test) 
# pf.find_weird_signals_noise(signal_loss=signal_loss,
#                             loss=1,
#                             noise_loss=noise_loss,
#                             signal=x_test[smask_test],
#                             limit=10**-2)  

####################                        ###############################
#################### Save new csv file copy ###############################
####################                        ###############################
# TODO uncomment
# folder_path='/home/halin/Autoencoder/Models/'
# save_path='/home/halin/Autoencoder/Models/mixed_models/'
# start_model = 172
# end_model = 173
# present_prefix = ''
# new_prefix = 'test_2_'
# for i in range(start_model, end_model):
#     #result_path = folder_path + f'CNN_{i}/' + present_prefix + 'results.csv'
#     result_path = '/home/halin/Autoencoder/Models/mixed_models/test_mixed_results.csv'
#     try:
#       results = pd.read_csv(result_path)
#     except OSError as e:
#       print(f'No file in folder CNN_{i}')
#       continue
#     new_results = results[['Model name', 'Epochs', 'Latent space', 'Number of filters']]
#     #save_path = folder_path + f'CNN_{i}/' + new_prefix + 'results.csv'
#     save_path = '/home/halin/Autoencoder/Models/mixed_models/' + new_prefix +'results.csv'
#     new_results.to_csv(save_path)      