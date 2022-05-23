import matplotlib.pyplot as plt
import pandas as pd
from creating_models import load_models

import plot_functions as pf
import data_manage as dm

import numpy as np
from tensorflow.keras.models import Sequential, load_model


############   Coolecting best models  ###################
#best_models = ['CNN_001_model_5', 'CNN_003_model_9', 'CNN_004_model_2', 'CNN_005_model_4', 'CNN_007_model_1', 'CNN_008_model_1', 'CNN_009_model_2']
#path='/home/halin/Autoencoder/Models/'
#save_path = path + 'Best_models' + '/best_results.csv'
#df_of_best_models(best_models=best_models,save_path=save_path,path=path)
#plot_table(path + 'Best_models', table_name='best_results.csv', headers=['Model name', 'Epochs', 'Batch', 'Kernel', 'Learning rate', 'True pos.', 'Latent space', 'Flops', 'Layers'])

############  Save best model in an folder  ###################
# best_row_model = results.loc[results['Model name'] == 'CNN_003_model_9']
# best_row_model.to_csv(path + 'Best_models' + '/best_model.csv')


###########  Create noise reduction curv incl best model  ##############
# best_model_path = '/home/halin/Autoencoder/Models/Best_models/best_model.csv'
# best_model = pd.read_csv(best_model_path)
# load_path = '/home/halin/Autoencoder/Models/CNN_003'
# save_path ='/home/halin/Autoencoder/Models/test_models' 
# pf.noise_reduction_from_results(pd.read_csv(load_path + '/results.csv'), x_low_lim=0.8, save_path= save_path, name_prefix='Incl_best_model_', best_model=best_model )

###########  Load a model and look at the structure  ###########
path = '/home/halin/Autoencoder/Models/CNN_003/CNN_003_model_9.h5'
model = load_model(path)
x_test, y_test, smask_test, signal, noise, std, mean = dm.load_data()
x_train, smask_train, y_train = dm.create_data(signal,noise)


##########   Test performance  #########
x_noise = x_test[~smask_test]
x_pred_noise = model.predict(x_noise)[0]
x_noise = dm.unnormalizing_data(x_noise[0], std=std, mean=mean)
x_pred_noise = dm.unnormalizing_data(x_pred_noise, std=std, mean=mean)

x_signal = x_test[smask_test]
x_pred_signal = model.predict(x_signal)[0]
x_signal = dm.unnormalizing_data(x_signal[0], std=std, mean=mean)
x_pred_signal = dm.unnormalizing_data(x_pred_signal, std=std, mean=mean)
plt.plot(x_noise[0], label="Original noise")
plt.plot(x_pred_noise, label="Predicted noise")
plt.legend()
plt.savefig('/home/halin/Autoencoder/Models/test_models/Noise_and_pred_noise')
plt.cla()
plt.plot(x_signal[0], label="Original signal")
plt.plot(x_pred_signal, label="Predicted signal")
plt.legend()
plt.savefig('/home/halin/Autoencoder/Models/test_models/Signal_and_pred_signal')
# print(model.summary())
#pf.plot_table()