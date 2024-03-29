### Tensorflow 2.2  ###########
import os
from gpuutils import GpuUtils
GpuUtils.allocate(gpu_count=1, framework='keras')
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
import sys

import matplotlib.pyplot as plt
import pandas as pd


import numpy as np
from tensorflow.keras.models import load_model
from tensorflow import keras
from contextlib import redirect_stdout

import Help_functions.creating_models as cm
import Help_functions.plot_functions as pf
import Help_functions.data_manage as dm

# from NewPhysicsAutoencoder import NewPhysicsAutoencoder
# from SecondCNNModel import SecondCNNModel
# from DenseModel import DenseModel
# from ConvAutoencoder import ConvAutoencoder
############   Coolecting best models  ###################
# best_models = ['CNN_118_model_1', 'CNN_118_model_2', 'CNN_118_model_5', 'CNN_118_model_6', 'CNN_118_model_11', 'CNN_118_model_12'  ]
# path='/home/halin/Autoencoder/Models/'
# save_path = path + 'mixed_models' + '/mixed_results.csv'
# dm.make_dataframe_of_collections_of_models(best_models=best_models,save_path=save_path,path=path)
# pf.plot_table(path + 'mixed_models', table_name='mixed_results.csv', headers=['Model name', 'Epochs', 'Batch', 'Kernel', 'Learning rate', 'True pos.', 'Latent space','Act. last layer', 'Flops', 'Layers'])
# results = pd.read_csv(save_path)
# pf.noise_reduction_from_results(results=results, best_model='',x_low_lim=0.8, save_path=path + 'Mixed_models')

############  Save best model in an folder  ###################
# best_row_model = results.loc[results['Model name'] == 'CNN_003_model_9']
# best_row_model.to_csv(path + 'Best_models' + '/best_model.csv')


###########  Create noise reduction curv incl best model  ##############
# best_model_path = '/home/halin/Autoencoder/Models/Best_models/best_model.csv'
# best_model = pd.read_csv(best_model_path)
# load_path = '/home/halin/Autoencoder/Models/CNN_003'
# save_path ='/home/halin/Autoencoder/Models/test_models' 
# pf.noise_reduction_from_results(pd.read_csv(load_path + '/results.csv'), x_low_lim=0.8, save_path= save_path, name_prefix='Incl_best_model_', best_model=best_model )



############  Add models to a dataframe   ##############

# model_names = ['CNN_101_model_5', 'CNN_003_model_6']
# main_path = '/home/halin/Autoencoder/Models'
# results = dm.create_dataframe_of_results(path=main_path,model_names=model_names)
# result_path = '/home/halin/Autoencoder/Models/mixed_models'
# csv_result_path = result_path + '/collected_results.csv'
# results.to_csv(csv_result_path)
# pf.plot_table(path=result_path, table_name='collected_results.csv' )
# pf.noise_reduction_from_results(pd.read_csv(result_path + '/collected_results.csv'), x_low_lim=0.8, save_path= result_path, name_prefix='', best_model='' )

###########  Model summary ##########################
save_path = '/home/halin/Autoencoder/Models/test_models'
model_path = '/home/halin/Autoencoder/Models/Wrong trained models/CNN_003/CNN_003_model_9.h5'
model = load_model(model_path)
print(model.summary())
# for i in range(1,7):

#   model_path = f'/home/halin/Autoencoder/Models/CNN_104/CNN_104_model_{i}.h5'
#   model = load_model(model_path)
#   print(model.summary())
#   next_model = input("Next model: \n")

#############  Plot table #########################

#pf.plot_table('/home/halin/Autoencoder/Models/CNN_107', headers=['Model name', 'Epochs','Act. last layer', 'Flops'], )

###########  Plot model summary   #############3
# number_of_models = 7
# folder = 'CNN_102'
# for i in range(number_of_models):
#     path = '/home/halin/Autoencoder/Models/' + folder +'/'+ folder + f'_model_{i+1}.h5'
#     model = load_model(path)
#     print(model.summary())
#     with open(path[:-2] + '_summary.txt', 'w') as f:
#         with redirect_stdout(f):
#             model.summary()

############  testing models  ###################
# data_url = '/home/halin/Autoencoder/Data/'
# x_test, y_test, smask_test, signal, noise, std, mean = dm.load_data(all_signals=False, data_path=data_url, small_test_set=1000)
# (encoder, decoder, autoencoder) = DenseModel.build(data=x_test,
#                                                      filters=[50,25], 
#                                                      activation_function='relu',
#                                                      latent_size=2,
#                                                      kernel=3,
#                                                      last_activation_function='linear',
#                                                       )#convs=1
# adam = keras.optimizers.Adam(learning_rate=0.0001) 
# autoencoder.compile(
#       loss = 'mse',
#       optimizer = adam,
#       metrics = ['mse','mae','mape'] )
# print(autoencoder.summary())                                                       

