import numpy as np

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

def load_data(all_signals=True, data_path='/home/halin/Autoencoder/Data/'):
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

  for i in range(1,10):
    noise = np.vstack((noise,np.load(data_path + f'trimmed100_data_noise_3.6SNR_1ch_000{i}.npy')))

  noise = np.vstack((noise,np.load(data_path + 'trimmed100_data_noise_3.6SNR_1ch_0010.npy')))
  signal = np.load(data_path + "trimmed100_data_signal_3.6SNR_1ch_0000.npy")
  signal = np.vstack((signal,np.load(data_path + "trimmed100_data_signal_3.6SNR_1ch_0001.npy")))
  n_classes = 2
 
  noise, signal, std, mean = normalizing_data(signal, noise)

  shuffle = np.arange(noise.shape[0], dtype=np.int)
  np.random.shuffle(shuffle)
  noise = noise[shuffle]
  shuffle = np.arange(signal.shape[0], dtype=np.int)
  np.random.shuffle(shuffle)
  signal = signal[shuffle]

  number_of_test_samples = 0
  if all_signals:
    number_of_test_samples = len(signal)
  else:  
    number_of_test_samples = 20000

  

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

  return x_test, y_test, smask_test, signal, noise, std, mean 

def create_data(signal, noise, signal_ratio=0.001, test_run=False ):
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

  if test_run:
    noise_train = noise[:mini_batch_size]
    signal_train = signal[:mini_batch_size]
  else:  
    noise_train = noise
    number_of_train_signals = np.floor((number_of_noise_samples)*signal_ratio).astype(int)
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
  

