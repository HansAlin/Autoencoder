import numpy as np
import pandas as pd
import Help_functions.creating_models as cm
import Help_functions.plot_functions as pf
import glob


def normalizing_data(signal, noise):
  """
    This function normalize the data using mean and standard
    deviation from noise data
  """
  std = np.std(noise)
  mean = np.mean(noise)
  normalized_noise = (noise - mean)/std
  normalized_signal = (signal - mean)/std
  
  return normalized_noise, normalized_signal, std, mean

def unnormalizing_data(normalized_data, std, mean):
  data = normalized_data*std + mean
  return data

def load_data(all_signals_for_testing=True, all_samples=True, data_path='/home/halin/Autoencoder/Data/', small_test_set=1000, number_of_files=10):
  """
    This function loads data from ARIANNA group, downloaded localy
    Args:
     all_signals = True means that all the signals are
    used in the test data. If all_signals = False only 20000 signals are used as test data.
    Can be useful if training on signals aswell or just want to test data on small
    test data.
    Returns:
      x_test, y_test, smask_test, signal, noise, std, mean
    
  """
  DATA_URL = data_path + 'trimmed100_data_noise_3.6SNR_1ch_0000.npy'#/home/halin/Autoencoder/Data/trimmed100_data_noise_3.6SNR_1ch_0000.npy
  noise = np.load(DATA_URL)

  for i in range(1,number_of_files):
    noise = np.vstack((noise,np.load(data_path + f'trimmed100_data_noise_3.6SNR_1ch_000{i}.npy')))

  noise = np.vstack((noise,np.load(data_path + 'trimmed100_data_noise_3.6SNR_1ch_0010.npy')))
  signal = np.load(data_path + "trimmed100_data_signal_3.6SNR_1ch_0000.npy")
  signal = np.vstack((signal,np.load(data_path + "trimmed100_data_signal_3.6SNR_1ch_0001.npy")))
  #n_classes = 2
 
  noise, signal, std, mean = normalizing_data(signal, noise)
  
  shuffle = np.arange(noise.shape[0], dtype=np.int)
  np.random.shuffle(shuffle)
  noise = noise[shuffle]
  shuffle = np.arange(signal.shape[0], dtype=np.int)
  np.random.shuffle(shuffle)
  signal = signal[shuffle]

  number_of_test_samples = 0
  if all_samples:
    number_of_test_samples = len(signal)
  else:  
    number_of_test_samples = small_test_set

  if all_signals_for_testing:
    signal_test = signal[:number_of_test_samples]
    noise_test = noise[:number_of_test_samples*2]
  else:
    number_of_test_samples = np.floor(number_of_test_samples/2).astype(int)
    signal_test = signal[:number_of_test_samples]
    noise_test = noise[:number_of_test_samples*2]  
  

  
  
  signal = signal[number_of_test_samples:]
  noise = noise[number_of_test_samples*2:]

  x_test = np.vstack((noise_test, signal_test))
  x_test = np.expand_dims(x_test, axis=-1)
  y_test = np.ones(len(x_test))
  y_test[:len(noise_test)] = 0
  shuffle = np.arange(x_test.shape[0])  #, dtype=np.int
  np.random.shuffle(shuffle)
  x_test = x_test[shuffle]
  y_test = y_test[shuffle]
  smask_test = y_test == 1

  #### Just for creating plot examples runs only ones ############
  # plot_signal_1 = x_test[smask_test][100]
  # plot_signal_2 = x_test[smask_test][110]
  # plot_noise_1 = x_test[~smask_test][100]
  # plot_noise_2 = x_test[~smask_test][110]
  # plot = np.hstack((plot_signal_1,plot_signal_2))
  # plot = np.hstack((plot, plot_noise_1))
  # plot = np.hstack((plot, plot_noise_2))
  # np.save('/home/halin/Autoencoder/Data/plot_examples.npy', plot)

  return x_test, y_test, smask_test, signal, noise, std, mean 

def create_data(signal, noise, signal_ratio=0, test_run=False, maximum_ratio=0.1 ):
  """
    This function creates training(validation) and test data based on choosen 
    signal ratio in sample.
    Args:
      signal = samples with signal
      noise = samples with noise
      test_run = creates a small training batch just for testing rest of code
    Returns:
      x_train, smask_train, y_train
      
  """
  
  mini_batch_size = 10000
  number_of_noise_samples = np.size(noise[:,0])
  number_of_signal_samples = np.size(signal[:,0])

  if test_run:
    noise_train = noise[:mini_batch_size]
    signal_train = signal[:mini_batch_size]
  else: 
    if (number_of_signal_samples and maximum_ratio) != 0: 
      number_of_train_noise = np.floor(number_of_signal_samples/maximum_ratio).astype(int)
      noise_train = noise[:number_of_train_noise]
    else:
      noise_train = noise[:number_of_noise_samples]
      
    if signal_ratio > 0:
      number_of_train_signals = np.floor((number_of_train_noise) / (1/signal_ratio - 1)).astype(int)
    else:
      number_of_train_signals = 0
    signal_train = signal[:number_of_train_signals]



  x_train = np.vstack((noise_train, signal_train))
  x_train = np.expand_dims(x_train, axis=-1)
  y_train = np.ones(len(x_train))
  y_train[:len(noise_train)] = 0
  shuffle = np.arange(x_train.shape[0]) #, dtype=np.int
  np.random.shuffle(shuffle)
  x_train = x_train[shuffle]
  y_train = y_train[shuffle]
  smask_train = y_train == 1

  

  return x_train, smask_train, y_train
  
def adding_noisereduction_values_to_result_table(load_path, save_path, x_test, smask_test):
	"""
		Add values to the result table
	"""
	#TODO check that pf.noise.reduction_curve_multi_models
	models = cm.load_models(load_path)
	read_results_path = load_path + '/' + 'results.csv'
	results = pd.read_csv(read_results_path)

	noise_reduction_factors = []
	true_pos_arrays = []
	for i, model in enumerate(models):
		threshold_value, tpr, fpr, tnr, fnr, noise_reduction_factor, true_pos = pf.noise_reduction_curve_multi_models([model],load_path, fpr=0.05, x_test=x_test, smask_test=smask_test,  save_outputs=False)

		results.loc[[i], ['True pos.']] = tpr
		results.loc[[i], ['False pos.']] = fpr

		noise_reduction_factors.append(noise_reduction_factor)
		true_pos_arrays.append(true_pos)

	
	results['Noise reduction'] = noise_reduction_factors
	results['True pos. array'] = true_pos_arrays

	results = results[['Model name','Epochs','Batch','Kernel','Learning rate','Signal ratio','False pos.','True pos.','Threshold value','Latent space','Number of filters','Flops','Layers','Noise reduction','True pos. array']]
	
	results.to_csv(save_path)

def make_dataframe_of_collections_of_models(best_models, save_path, path='/home/halin/Autoencoder/Models/', prefix=''):
  
  best_results = pd.DataFrame(columns=[ 'Model name', 'Epochs', 'Batch', 'Kernel', 'Learning rate', 'Signal ratio', 'False pos.', 'True pos.', 'Threshold value', 'Latent space', 'Number of filters', 'Flops', 'Layers', 'Noise reduction','True pos. array'])
  for i, model in enumerate(best_models):
    folder = model[:7]
    # TODO remove next two lines
    if folder == '171_mod':
      print('error')
      
    try:
      results = pd.read_csv(path + folder + '/' + prefix + 'results.csv')
    except OSError as e:
      print(f'No file in folder CNN_{folder}')
      continue  
    if model in results.values:
      best_row_model = results.loc[results['Model name'] == model]
      best_results = best_results.append(best_row_model)
    
  
  best_results.to_csv(save_path)

def create_dataframe_of_results( model_names, path='/home/halin/Autoencoder/Models'):
  results_to_return = ''
  first_result = True
  list_of_results_csv = glob.glob(path + '/**/*' + '.csv', recursive=True)
  for csv_result_path in list_of_results_csv:
    last_part = csv_result_path[-12:] 
    if last_part == '/results.csv':
      results = pd.read_csv(csv_result_path)
      for model_name in model_names:
        model_result = results.loc[results['Model name'] == model_name]
        if first_result:
          if  not model_result.empty:
            results_to_return = model_result
            first_result = False
        else:
          if  not model_result.empty:
            results_to_return = results_to_return.append(model_result)  
  return results_to_return        