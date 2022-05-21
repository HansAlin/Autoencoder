from matplotlib.pyplot import axis
import pandas as pd
from creating_models import load_models
from plot_functions import plot_table, noise_reduction_from_results
from data_manage import df_of_best_models
import numpy as np



#best_models = ['CNN_001_model_5', 'CNN_003_model_9', 'CNN_004_model_2', 'CNN_005_model_4', 'CNN_007_model_1', 'CNN_008_model_1', 'CNN_009_model_2']
#path='/home/halin/Autoencoder/Models/'
#save_path = path + 'Best_models' + '/best_results.csv'

#df_of_best_models(best_models=best_models,save_path=save_path,path=path)
#plot_table(path + 'Best_models', table_name='best_results.csv', headers=['Model name', 'Epochs', 'Batch', 'Kernel', 'Learning rate', 'True pos.', 'Latent space', 'Flops', 'Layers'])
best_model_path = '/home/halin/Autoencoder/Models/Best_models/best_model.csv'
best_model = pd.read_csv(best_model_path)
path = '/home/halin/Autoencoder/Models/test_models/'
# best_row_model = results.loc[results['Model name'] == 'CNN_003_model_9']
# best_row_model.to_csv(path + 'Best_models' + '/best_model.csv')
noise_reduction_from_results(pd.read_csv(path + 'results.csv'), x_low_lim=0.8, save_path= path, name_prefix='', best_model=best_model )