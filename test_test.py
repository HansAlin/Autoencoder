import pandas as pd
from creating_models import load_models
import numpy as np

def df_of_best_models(best_models, save_path, path='/home/halin/Autoencoder/Models/'):
  
  best_results = pd.DataFrame(columns=[ 'Model name', 'Epochs', 'Batch', 'Kernel', 'Learning rate', 'Signal ratio', 'False pos.', 'True pos.', 'Threshold value', 'Latent space', 'Number of filters', 'Flops', 'Layers', 'Noise reduction','True pos. array'])
  for i, model in enumerate(best_models):
    folder = model[:7]
    results = pd.read_csv(path + folder + '/' + 'results.csv')
    best_row_model = results.loc[results['Model name'] == model]
    best_results.loc[i] = best_row_model
  
  best_results.to_csv(save_path)

best_models = ['CNN_001_model_2', 'CNN_001_model_4']
path='/home/halin/Autoencoder/Models/'
save_path = path + '/test_models' + '/best_results.csv'

df_of_best_models(best_models=best_models,save_path=save_path,path=path)