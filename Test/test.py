
from cProfile import label
from cmath import nan
from unittest import result
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
import matplotlib.image as mpimg
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


# ###########################    Create an encoder   ####################
# ###########################                        ####################
# ###########################                        ####################
data_url = '/home/halin/Autoencoder/Data/'
folder = 141
model_number = 9
[filters, latent_size] = pf.find_values_from_model(folder=folder,
                         model_number=model_number,
                         values_of_interest=['Number of filters', 'Latent space'])

x_test, y_test, smask_test, signal, noise, std, mean = dm.load_data(all_signals_for_testing=False,
                                                                all_samples=True,						
                                                                data_path=data_url, 
                                                                small_test_set=1000,
                                                                number_of_files=10)
model_path = f'/home/halin/Autoencoder/Models/CNN_{folder}/CNN_{folder}_model_{model_number}.h5'     
autoencoder = load_model(model_path)
saved_weights_path = '/home/halin/Autoencoder/Models/test_models/autoencoder_weights.h5'
autoencoder.save_weights(saved_weights_path, overwrite = True)
(encoder, decoder, autoencoder) = ConvAutoencoder.build(data=x_test,
                                                        filters=filters, 
                                                        activation_function='relu',
                                                        latent_size=latent_size,
                                                        kernel=3,
                                                        last_activation_function='linear' )
encoder.load_weights(saved_weights_path, skip_mismatch = True, by_name = True) 
encoder.compile(
      loss = 'mse',
      optimizer = 'adam',
      metrics = ['mse','mae','mape'] 
  )

encoder.summary() 
encoder.save('/home/halin/Autoencoder/Models/test_models/encoder.h5')  
encoder = load_model('/home/halin/Autoencoder/Models/test_models/encoder.h5')
signal_pred_values = encoder.predict(x_test[smask_test]) 
noise_pred_values = encoder.predict(x_test[~smask_test]) 
number = 1000

# signal_loss, noise_loss = pf.loss_values_from_latent_space(signal_pred_values, noise_pred_values)
# pf.noise_reduction_curve_single_model(model_name=f'Model_{folder}_model_{model_number}',
#                                     fpr=0.95,
#                                     x_low_lim=0.8,
#                                     save_path='/home/halin/Autoencoder/Models/test_models/Encoder_loss',
#                                     signal_loss=signal_loss,
#                                     noise_loss=noise_loss,
#                                     ) 
# ax1 = plt.hist(noise_loss, bins=100, label='Noise', color='blue', alpha=0.5)
# ax2 = plt.hist(signal_loss, bins=100, label='Signal', color='red', alpha=0.5)

# plt.scatter([[1,2,3,4,5,6,7]]*number,signal_pred_values[:number], color='green', label='Signals', alpha=0.5)
signal_pred_values =  np.transpose(signal_pred_values)
noise_pred_values = np.transpose(noise_pred_values)
plt.scatter(signal_pred_values[0], signal_pred_values[1], color='blue',label='Signal', alpha=0.1)

plt.scatter(noise_pred_values[0],noise_pred_values[1], color='red', label='Noise', alpha=0.8) 

plt.xlim(-5,5)
plt.ylim(-5,5)
plt.grid()
plt.legend()
plt.savefig('/home/halin/Autoencoder/Models/test_models/test_encoder_plot_1.png')
plt.show()
plt.cla()                                                 



######################   Test new loss computation    ###############
######################  and if performance depends on ##############
######################  where the test data origins from  #################
# start_folder = 102
# end_folder = 192
# x_low_lim = 0.8
# no_testing_folders = [112,113,114,116,130,135]
# folder_path = '/home/halin/Autoencoder/Models/'
# number_of_tests = 1
# x_test, y_test, smask_test, signal, noise, std, mean = dm.load_data()
# for folder in range(start_folder, end_folder):
#   result_path = folder_path + f'CNN_{folder}/results.csv'
#   if folder in no_testing_folders:
#     continue

#   try:
#     results = pd.read_csv(result_path)
#   except OSError as e:
#     print(f'No file in folder CNN_{folder}')
#     continue
#   save_path = f'/home/halin/Autoencoder/Models/CNN_{folder}/'
#   for i in range(0,number_of_tests):  
    
#     prefix = f'loss_test_{i}_'  # to not interfere with existing data
#     result_path = pf.change_new_results(results=results,
#                     x_test=x_test,
#                     smask_test=smask_test, 
#                     prefix=prefix,
#                     folder_path=folder_path, 
#                     folder=folder) 
    
    
#     result_path = pf.find_best_model_in_folder(start_model=folder,
#                                             end_model=folder + 1, #exclusive	
#                                             number_of_models=20, 
#                                             terms_of_condition ='', #'', #Epochs
#                                             value_of_condition ='', #'', #tanh
#                                             comparison = 'equal',
#                                             x_low_lim = x_low_lim,
#                                             prefix= prefix,
#                                             result_path=result_path,
#                                             save_path=save_path,
#                                             headers=['Model name', 
#                                                     'Epochs', 
#                                                     'Number of filters',  
#                                                     #'Kernel', 
#                                                     #'Batch',
#                                                     #'Flops',
#                                                     'Latent space',
#                                                     #'Act. last layer' 
#                                                     ]) #'Activation func. rest'Act. last layer linear
    

#################### Find best model based on ####################
#################### reduction curve          ####################
####################                          ####################

#  Filter models with linera activation function in last layer

# prefix = 'test_0_'
# x_low_lim = 0.8
# start_model = 101
# end_model = 193
# save_path = f'/home/halin/Autoencoder/Models/mixed_models/' #CNN_{start_model}/'#
# #result_path = '/home/halin/Autoencoder/Models/CNN_187/results.csv'
# result_path = pf.find_best_model_in_folder(start_model=start_model,
# 							              end_model=end_model, #exclusive	
#                                         number_of_models=1000, 
#                                         terms_of_condition ='Act. last layer', #'', #'Learning rate', #Epochs
#                                         value_of_condition ='tanh', #'', #'tanh',#0.001, #
#                                         comparison = 'equal',
#                                         x_low_lim = x_low_lim,
#                                         save_path=save_path,
#                                         prefix=prefix,
#                                         #result_path=result_path,
#                                         headers=['Model name', 
#                                                 'Epochs', 
#                                                 'Number of filters',  
#                                                 'Kernel', 
#                                                 'Batch',
#                                                 'Flops',
#                                                 'Learning rate',
#                                                 'Signal ratio',
#                                                 'Latent space',
#                                                 'Act. last layer' 
#                                                 ]) #'Activation func. rest'Act. last layer linear

# Filter models with greater than 150 epochs                                    
# result_path = pf.find_best_model_in_folder(start_model=start_model,
#                             end_model=end_model,
#                             terms_of_condition='Epochs',
#                               value_of_condition=150,
#                               number_of_models=700,
#                               comparison='greater',
#                               result_path=result_path,
#                               save_path=save_path,
#                               x_low_lim=x_low_lim,
#                               prefix=prefix,
#                               headers=['Model name', 
#                                       'Epochs', 
#                                       'Number of filters',  
#                                       'Kernel', 
#                                       'Batch',
#                                       #'Flops',
#                                       'Latent space',
#                                       'Act. last layer',
#                                       ] )

### Learning rate
# result_path = pf.find_best_model_in_folder(start_model=start_model,
#                             end_model=end_model,
#                             terms_of_condition='Learning rate',
#                               value_of_condition=0.0001,
#                               number_of_models=700,
#                               comparison='equal',
#                               result_path=result_path,
#                               save_path=save_path,
#                               x_low_lim=x_low_lim,
#                               prefix=prefix,
#                               headers=['Model name', 
#                                       'Epochs', 
#                                       'Number of filters',  
#                                       #'Kernel', 
#                                       #'Batch',
#                                       #'Flops',
#                                       'Latent space',
#                                       #'Act. last layer',
#                                       ] )

### Layers
# result_path = pf.find_best_model_in_folder(start_model=start_model,
#                             end_model=end_model,
#                             terms_of_condition='Layers',
#                               value_of_condition=1,
#                               number_of_models=500,
#                               comparison='equal',
#                               result_path=result_path,
#                               save_path=save_path,
#                               x_low_lim=x_low_lim,
#                               prefix=prefix,
#                               headers=['Model name', 
#                                       'Epochs', 
#                                       'Number of filters',  
#                                       #'Kernel', 
#                                       #'Batch',
#                                       #'Flops',
#                                       'Latent space',
#                                       #'Act. last layer',
#                                       ] )
# # batch size                                      
# result_path = pf.find_best_model_in_folder(start_model=start_model,
#                             end_model=end_model,
#                             terms_of_condition='Batch',
#                               value_of_condition=1024,
#                               number_of_models=700,
#                               comparison='equal',
#                               result_path=result_path,
#                               save_path=save_path,
#                               x_low_lim=x_low_lim,
#                               prefix=prefix,
#                               headers=['Model name', 
#                                       'Epochs', 
#                                       'Number of filters',  
#                                       #'Kernel', 
#                                       #'Batch',
#                                       #'Flops',
#                                       'Latent space',
#                                       #'Act. last layer',
#                                       ] )

# # #kernel size
# result_path = pf.find_best_model_in_folder(start_model=start_model,
#                             end_model=end_model,
#                             terms_of_condition='Kernel',
#                               value_of_condition=3,
#                               number_of_models=10,
#                               comparison='equal',
#                               result_path=result_path,
#                               save_path=save_path,
#                               x_low_lim=x_low_lim,
#                               prefix=prefix,
#                               headers=['Model name', 
#                                       'Epochs', 
#                                       'Number of filters',  
#                                       #'Kernel', 
#                                       #'Batch',
#                                       #'Flops',
#                                       'Latent space',
#                                       #'Act. last layer',
#                                       ] )
#Latent size
result_path = pf.find_best_model_in_folder(start_model=start_model,
                            end_model=end_model,
                            terms_of_condition='Latent space',
                              value_of_condition=2,
                              number_of_models=20,
                              comparison='equal',
                              result_path=result_path,
                              save_path=save_path,
                              x_low_lim=x_low_lim,
                              prefix=prefix,
                              headers=['Model name', 
                                      'Epochs', 
                                      'Number of filters',  
                                      'Kernel', 
                                      'Batch',
                                      #'Flops',
                                      'Latent space',
                                      'Act. last layer',
                                      ] )                                  
                                

####################                           ######################
####################     Perfomance plots      ######################
####################                           ######################

# x_test, y_test, smask_test, signal, noise, std, mean = dm.load_data()
# std = 0.011491077671030752
# mean = 2.6521230839856967e-08
# plot_examples = np.load('/home/halin/Autoencoder/Data/plot_examples.npy')
# save_plot_path = '/home/halin/Autoencoder/Pictures/'
# title = ' Model ([32,16,8,4],[4])'
# start_folder = 171
# end_folder = 172
# model = 1
# for folder in range(start_folder, end_folder):

    
#     folder_path = '/home/halin/Autoencoder/Models/'
#     #folder_path = '/home/halin/Autoencoder/Models/Wrong trained models/'
#     if folder < 10:
#         folder =  f'CNN_00{folder}'
#     elif folder < 100:
#         folder = f'CNN_0{folder}'
#     else:       
#         folder = f'CNN_{folder}'
#     try:

#       results = pd.read_csv(folder_path + folder + '/results.csv')
#     except OSError as e:
#       print(f'No file in folder {folder}')
#       continue
#     (rows, cons) = results.shape
    
#     for model_number in range(1,rows + 1):
#         save_path = folder_path + folder + '/' + folder + f'_model_{model_number}'
#         if model != '':
#             rows = 0
#             model_number = model
            
#         else:     
#             save_plot_path = save_path
        
#         # And here
#         model_path = save_path + '.h5'
#         try:
#           model = load_model(model_path) 
#         except OSError as e:
#           print(f'No model {model_number}')
#           continue
#         sufix = 1
#         to_plot = np.vstack((plot_examples[:,0], plot_examples[:,2]))
#         pf.plot_single_performance(model,to_plot,save_plot_path,std,mean, sufix=sufix, plot_title=title)
#         plt.cla()
#         sufix = 2
#         to_plot = np.vstack((plot_examples[:,1], plot_examples[:,3]))
#         pf.plot_single_performance(model,to_plot,save_plot_path,std,mean, sufix=sufix, plot_title=title)
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
# x_test, y_test, smask_test, signal, noise, std, mean = dm.load_data(all_signals_for_testing=False,
#                                                                     small_test_set= 1000    )
# model_path = '/home/halin/Autoencoder/Models/CNN_194/CNN_194_model_1.h5'
# save_path = '/home/halin/Autoencoder/Models/test_models/'
# try:
#     model = load_model(model_path)
# except:
#     print('Could not find model!') 

# signal_loss , noise_loss = pf.costum_loss_values_3(model=model,x=x_test,smask=smask_test)
# pf.noise_reduction_curve_single_model("Big model",
#                                  save_path=save_path,
#                                  fpr=0.05,
#                                  signal_loss=signal_loss,
#                                  noise_loss=noise_loss)


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
# 
# 
####################                        ###############################
#################### Adjust loss values     ###############################
####################                        ###############################  
# def adjust_loss(x):
#     number_of_models = len(x)
#     for i in range(0,number_of_models):
#         x[i] = pf.convert_result_dataframe_string_to_list(x[i])
#         if len(x[i]) == 243188:
#             return x[i]*243188/100
#         else:
#             return x[i]*121594/100

# def reconstruct_loss_values(noise_reduction_factor, true_pos):
#     bins = len(noise_reduction_factor)
    
#     true_neg = [0]*bins
#     false_neg = [0]*bins
#     false_pos = [0]*bins
#     for i in range(bins):
#         if noise_reduction_factor == 243188:
#             true_neg[i] = 1
#         else:
#             true_neg[i] = 1 - 1/noise_reduction_factor[i]   
#         false_neg[i] = 1 - true_pos[i]
#         false_pos[i] = 1 - true_neg[i]
#     return 0    


# folder_path='/home/halin/Autoencoder/Models/'

# start_model = 176
# end_model = start_model + 1
# present_prefix = ''
# new_prefix = 'test_3_'
# for i in range(start_model, end_model):
#     result_path = folder_path + f'CNN_{i}/' + present_prefix + 'results.csv'
#     save_path_reduction_curve=f'/home/halin/Autoencoder/Models/CNN_{i}/' + new_prefix
#     try:
#       results = pd.read_csv(result_path)
#     except OSError as e:
#       print(f'No file in folder CNN_{i}')
#       continue
#     loss_values = results['Signal loss'].values[0]
#     noise_reduction_values = results['Noise reduction'].values[0]
#     results['Signal loss'] = results['Signal loss'].apply(adjust_loss)
#     results['Noise loss'] = results['Noise loss'].apply(adjust_loss) 
#     pf.noise_reduction_from_results(results=results,
#                                 save_path=save_path_reduction_curve,
#                         )
#     # TODO make a noise reduction plot from results
#     save_path = folder_path + f'CNN_{i}/' + new_prefix + 'results.csv'
    
#     results.to_csv(save_path)  

####################                        ###############################
#################### Remove unwanted files  ###############################
####################                        ###############################  
# for folder in range(109,182):
#   remove_from_dir = f'/home/halin/Autoencoder/Models/CNN_{folder}'
#   for file_name in os.listdir(remove_from_dir):
#     if file_name.startswith('test'):
#       remove_file_name = remove_from_dir + '/' +file_name
#       os.remove(remove_file_name)

####################                        ###############################
#################### Plot noise reduction   ###############################
####################     and tabel          ###############################

# start_model = 188
# end_model = 190
# file_prefix = ''
# x_low_lim = 0.75
# for model in range(start_model, end_model):
#   save_path = f'/home/halin/Autoencoder/Models/CNN_{model}/'
#   result_path  = save_path + file_prefix + 'results.csv'
#   try:
#     results = pd.read_csv(result_path)
#   except OSError as e:
#     print(f'No file in folder {result_path}')
#     continue
#   pf.noise_reduction_from_results(results=results,
#                                   best_model='',
#                                   x_low_lim=x_low_lim,
#                                   save_path=save_path,
#                                   name_prefix=file_prefix) 
#   pf.plot_table(results=results,
#                 save_path=save_path,
#                 headers=['Model name', 
#                         'Learning rate',
#                         'Kernel'],
#                         )                                

####################                        ###############################
#################### New models based on    ###############################
#################### previous hyperparameters  ###############################

# result_path = '/home/halin/Autoencoder/Models/mixed_models/test_0_sorted_results.csv'
# save_path = '/home/halin/Autoencoder/Models/CNN_998/'
# pf.create_models_from_data_frame(result_path=result_path, 
#                       path=save_path,
#                       plot=True)  

# x_test, y_test, smask_test, signal, noise, std, mean = dm.load_data()
# pf.integration_test(plot=True, x_test=x_test,
#              smask_test=smask_test, 
#              save_path='/home/halin/Autoencoder/Models/test_models/')   

# model = load_model('/home/halin/Autoencoder/Models/CNN_999/CNN_158_model_1.h5')
# x_test, y_test, smask_test, signal, noise, std, mean = dm.load_data() 
# pf.costum_loss_values(model=model, x=x_test, smask=smask_test)     

# results = pd.read_csv('/home/halin/Autoencoder/Models/CNN_999/results.csv')
# save_path = '/home/halin/Autoencoder/Models/mixed_models/'
# pf.noise_reduction_from_results(results=results, save_path=save_path, best_model='')  


#############################                         ############################
############################# Plot 3 reduction curves ############################
#############################                         ############################
# test_size = 1000
# x_test, y_test, smask_test, signal, noise, std, mean = dm.load_data()
# save_path = '/home/halin/Autoencoder/Pictures/'+'Best_mosel_Signal_efficiency_vs_noise_reduction_factor.png'

# model = load_model('/home/halin/Autoencoder/Models/CNN_171/CNN_171_model_1.h5')
# signal_loss, noise_loss = pf.costum_loss_values(model, x_test,smask_test)
# threshold_value, tpr, fpr, tnr, fnr, nrf, tp = pf.noise_reduction_curve_single_model("Model",
#                 '',
#                 0.05,
#                 signal_loss,
#                 noise_loss,
#                 False)
# plt.plot(tp,nrf, label='Best model')

# signal_loss, noise_loss = pf.costum_loss_values_4(x=x_test,
#              smask=smask_test)  
# threshold_value, tpr, fpr, tnr, fnr, nrf, tp = pf.noise_reduction_curve_single_model("Name", 
#                 save_path='', 
#                 fpr=0.05, 
#                 signal_loss=signal_loss, 
#                 noise_loss=noise_loss, 
#                 plot=False)      
# plt.plot(tp,nrf, label='Baseline')

# model = load_model('/home/halin/Autoencoder/Models/CNN_164/CNN_164_model_2.h5')
# signal_loss, noise_loss = pf.costum_loss_values(model, x_test,smask_test)
# threshold_value, tpr, fpr, tnr, fnr, nrf, tp = pf.noise_reduction_curve_single_model("Model",
#                 '',
#                 0.05,
#                 signal_loss,
#                 noise_loss,
#                 False)
# plt.plot(tp,nrf, label='Worst model') 

# model = load_model('/home/halin/Autoencoder/Models/CNN_194/CNN_194_model_1.h5')
# signal_loss, noise_loss = pf.costum_loss_values(model, x_test,smask_test)
# threshold_value, tpr, fpr, tnr, fnr, nrf, tp = pf.noise_reduction_curve_single_model("Model",
#                 '/home/halin/Autoencoder/Models/test_models/test_4_',
#                 0.05,
#                 signal_loss,
#                 noise_loss,
#                 False)
# plt.plot(tp,nrf, label='Big model') 

# noise_events = len(noise_loss)
# plt.legend()
# plt.ylabel(f'Noise reduction factor. Total {noise_events} noise events')
# plt.xlabel('Efficiency/True Positive Rate')
# plt.title('Signal efficiency vs. noise reduction factor')
# plt.semilogy(True)
# plt.xlim(0.75,1)               
# plt.grid()  
# plt.tight_layout()
# plt.savefig(save_path)

#############################                         ############################
############################# Plot reduction curves ############################
#############################                         ############################
#  
# [[190,1],[190,2],[190,3],[190,4],[190,5]] # Signal ratio [0.00, 0.01, 0.02, 0.04, 0.08]
# model = [[187,2],[187,6],[187,10],[187,14]] #  
# label_title = 'Kernel size'
# labels = ['3','5', '7','9']
# prefix = '' # to result path e.g. test_, test_0_
# save_path = '/home/halin/Autoencoder/Pictures/' + label_title + ' SE_vs_NR.png'
# x_low_lim = 0.8

# pf.plot_several_NR_curves(model, labels,prefix,save_path,x_low_lim, label_title)


# pf.count_models()
# path = '/home/halin/Autoencoder/Models/CNN_189/results.csv'
# save_path = '/home/halin/Autoencoder/Models/CNN_189/'
# results = pd.read_csv(path)
# pf.noise_reduction_from_results(results=results,
#                             best_model='',
#                             save_path=save_path,
#                             )

# path = '/home/halin/Autoencoder/Models/'
# save_path = '/home/halin/Autoencoder/Models/plots/loss.png'
# for folder in range(101,190):
#   search_path = path + f'CNN_{folder}/'
#   result_path = search_path + '/results.csv'
#   try:
#     results = pd.read_csv(result_path)
#   except OSError as e:
#     print(f'No file in folder {result_path}')
#     continue 
#   for index, row in results.iterrows():
#     loss_pic_path = search_path + f'CNN_{folder}_model_1_loss_plot.png'
#     img = mpimg.imread(loss_pic_path)
#     imgplot = plt.imshow(img)
#     plt.savefig(save_path)

# x_test, y_test, smask_test, signal, noise, std, mean = dm.load_data()
# model = load_model('/home/halin/Autoencoder/Models/CNN_171/CNN_171_model_1.h5')
# signal_loss, noise_loss = pf.costum_loss_values(model=model,x=x_test,smask=smask_test)
# save_path = '/home/halin/Autoencoder/Pictures/Best_model'
# pf.hist(path=save_path,signal_loss=signal_loss,noise_loss=noise_loss,)

# model = load_model('/home/halin/Autoencoder/Models/CNN_195/CNN_195_model_1.h5')
# signal_loss, noise_loss = pf.costum_loss_values(model=model,x=x_test,smask=smask_test)
# save_path = '/home/halin/Autoencoder/Pictures/Big_model'
# pf.hist(path=save_path,signal_loss=signal_loss,noise_loss=noise_loss,)