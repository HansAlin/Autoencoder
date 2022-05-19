import data_manage as dm
import creating_models as cm
import pandas as pd 
from data_manage import load_data
from creating_models import create_and_train_model

# Hyper parameters
batches = [1024]
learning_rates = [10**(-4)]
signal_ratios = [0]
kernels = [3]
latent_spaces = [2]
number_of_filters = [8]
layers = [1]
epochs = 1

model_number = 1
test_run = True
all_signals = False
plot =True
fpr = 0.05
verbose = 1

x_test, y_test, smask_test, signal, noise, std, mean = load_data(all_signals=all_signals)
results = pd.DataFrame(columns=[ 'Model name', 'Epochs', 'Batch', 'Kernel', 'Learning rate', 'Signal ratio', 'False pos.', 'True pos.', 'Threshold value', 'Latent space', 'Number of filters', 'Flops', 'Layers', 'Noise reduction'])
path = '/home/halin/Autoencoder/Models/test_models'



for batch in batches:
  for learning_rate in learning_rates:
    for signal_ratio in signal_ratios:
      for kernel in kernels:
        for latent_space in latent_spaces:
          for filters in number_of_filters:
            for layer in layers:
              results.loc[model_number] = create_and_train_model(layers=layer,
                                                               model_number=model_number,
                                                              latent_space=latent_space,
                                                              test_run=test_run,
                                                              path=path,
                                                              signal=signal,
                                                              noise=noise,
                                                              verbose=verbose,
                                                              x_test=x_test,
                                                              smask_test=smask_test,
                                                              kernel=kernel,
                                                              epochs=epochs,
                                                              batch=batch,
                                                              learning_rate=learning_rate,
                                                              signal_ratio=signal_ratio, 
                                                              plot=plot,
                                                              fpr=fpr,
                                                              number_of_filters=filters)
              model_number += 1


results.to_csv(path + '/results.csv')