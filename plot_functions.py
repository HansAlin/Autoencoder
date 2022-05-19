from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from tensorflow.keras.models import load_model


def find_signal(model, treshold, x, smask, under_treshold=True):
  """
    This function steps trough the losses to find data tha are
    below or above a sertain treshold.
    Args:
      model: keras model
      treshold: (float) value to compare
      x: data to test
      smask: where the true signals are
      under_treshold: (bool)
    Returns:
      outliers: the data beyond threshold in an list

  """
  outliers = []
  for i in range(len(x)):
    x_pred = model.predict(np.array([x[i],]))
    test = x[i]
    pred_loss = keras.losses.mean_squared_error(x[i], x_pred)
    pred_loss = np.sum(pred_loss)/len(pred_loss)
    if under_treshold:
      if pred_loss < treshold:
        outliers.append(x[i])
        
    else:
      if pred_loss > treshold:
        outliers.append(x[i])  
  return outliers      

def plot_signal_nois(x,smask):
  for trace in x[smask][:2]:
    fig, ax = plt.subplots(1, 1)
    ax.plot(trace)
    fig.tight_layout()
    plt.show()
    # plot a few noise events
  for noise in x[~smask][:2]:
    fig, ax = plt.subplots(1,1)
    ax.plot(noise)
    fig.tight_layout()
    plt.show() 

def loss_plot(path, trained_autoencoder):
  loss = trained_autoencoder.history['loss']
  val_loss = trained_autoencoder.history['val_loss']

  epochs = range(len(loss))
  plt.figure()
  plt.plot(epochs, loss, 'bo', label='Training loss')
  plt.plot(epochs, val_loss, 'b', label='Validation loss')
  plt.title('Training and validation loss')
  plt.yscale('log')
  plt.legend()
  path = path + '_loss_plot.png'
  plt.savefig(path)
  plt.show()

  return 0

def plot_signal(x,smask):
  for item in x:
    fig, ax = plt.subplots(1,1)
    ax.plot(item)
    fig.tight_layout()
    plt.show()



  for trace in x[smask][:2]:
    fig, ax = plt.subplots(1, 1)
    ax.plot(trace)
    fig.tight_layout()
    plt.show()
    # plot a few noise events
  for noise in x[~smask][:2]:
    fig, ax = plt.subplots(1,1)
    ax.plot(noise)
    fig.tight_layout()
    plt.show() 

def prep_loss_values(model, x, smask):
  """
    This function predict the value using keras.predict and
    calculates the mse for signals and noise events. Add the values 
    and divide by sample size
    Args:
      model: keras model
      x: the test data shape (Any, 100, 1)
      smask: smask for x 
    Returns: 
      signal_loss, noise_loss 
  """
  data_bins = np.size(x[0])
  x_noise = x[~smask]
  x_pred_noise = model.predict(x_noise)
  x_signal = x[smask]
  x_pred_signal = model.predict(x_signal)
  noise_loss = keras.losses.mean_squared_error(x_noise, x_pred_noise)
  noise_loss = np.sum(noise_loss, axis=1)/data_bins     #Per sample bin

  signal_loss = keras.losses.mean_squared_error(x_signal, x_pred_signal)
  signal_loss = np.sum(signal_loss, axis=1)/data_bins

  return signal_loss, noise_loss

def hist(path, signal_loss, noise_loss, resolution=100, plot=True):
  
  max_value = np.max(signal_loss)
  min_value = np.min(noise_loss)
  low_lim = np.floor(np.log10(min_value))
  high_lim = np.floor(np.log10(max_value))
  bins = np.logspace(low_lim,high_lim , resolution)

  if plot:
    
    ax1 = plt.hist(noise_loss, bins=bins, log=True, alpha=0.5, density=True)
    ax2 = plt.hist(signal_loss, bins=bins, log=True, alpha=0.5, density=True)
    plt.xscale('log')
    plt.xlabel('Mean squared error')
    plt.ylabel('Counts')
    path = path + '_hist.png'
    plt.savefig(path)
    plt.show()
    
    
  return bins

def roc_curve(path, signal_loss, noise_loss, bins,fpr=0.05, plot=True):
  """
    This function takes signal and noise loss as arguments. They are 
    arrays from mse calculating.
    Bins is taken from hist
    Args: 
      path: where the plots saves
      signal_loss: calculated signal loss
      noise_loss: noise loss
      bins: number of bins to split the data
      fpr: False Positive Rate 
    Returns:
      thershold: value for a specific False Positive Ratio fpr
      tpr: True positive ratio 
      fpr: False positive ratio
      tnr: True negative ratio
      fnr: False negative ratio

  """
  min_value = np.min(signal_loss)
  max_value = np.max(signal_loss)

  thresholds = bins
  threshold_value = 0
  true_pos = np.zeros(len(bins))
  false_pos = np.zeros(len(bins))
  i = 0

  tpr = 0
  for limit in thresholds:
    true_pos[i] = np.count_nonzero(signal_loss > limit)/len(signal_loss)
    false_pos[i] =np.count_nonzero(noise_loss > limit)/len(noise_loss)
    if false_pos[i-1] > fpr :
      threshold_value = limit
      tpr = true_pos[i]

    i += 1

  fnr = 1 - tpr
  tnr = 1 - fpr  
  

  if plot:
    plt.plot(false_pos,true_pos)  
    plt.ylabel('True positive rate')
    plt.xlabel('False positive rate')
    plt.title('ROC')

    ## Vertical line at False Positive Rate limit
    y = np.linspace(0,1,2)
    x = [fpr]*2
    plt.plot(x,y)

    plt.grid()
    path = path + '_roc.png'
    plt.savefig(path)
    plt.show()

  return threshold_value, tpr, fpr, tnr, fnr

def noise_reduction_curve(path, signal_loss, noise_loss, fpr=0.05, plot=True):
  """
    This function takes signal and noise loss as arguments. They are 
    arrays from mse calculating.
    Bins is taken from hist
    Args: 
      path: where the plots saves
      signal_loss: calculated signal loss
      noise_loss: noise loss
      fpr: False Positive Rate 
    Returns:
      thershold: value for a specific False Positive Ratio fpr
      tpr: True positive ratio 
      fpr: False positive ratio
      tnr: True negative ratio
      fnr: False negative ratio

  """
  max_value = np.max(signal_loss)
  min_value = np.min(noise_loss)
  low_lim = np.floor(np.log10(min_value))
  high_lim = np.floor(np.log10(max_value))
  bins = np.logspace(low_lim,high_lim , 1000)

  
  threshold_value = 0
  true_pos = np.zeros(len(bins))
  false_pos = np.zeros(len(bins))
  true_neg = np.zeros(len(bins))
  false_neg = np.zeros(len(bins))
  noise_reduction_factor = np.zeros(len(bins))


  tpr = 0
  for i, limit in enumerate(bins):
    
    true_pos[i] = np.count_nonzero(signal_loss > limit)/len(signal_loss)
    false_pos[i] =np.count_nonzero(noise_loss > limit)/len(noise_loss)
    true_neg[i] = 1 - false_pos[i]
    false_neg[i] = 1 - true_pos[i]
    

    if (true_neg[i] < 1):
      noise_reduction_factor[i] = 1 / ( 1 - true_neg[i])
    else:
      noise_reduction_factor[i] = len(noise_loss)  
    
    
    if false_pos[i] > fpr:
      threshold_value = limit
      tpr = true_pos[i]


  fnr = 1 - tpr
  tnr = 1 - fpr  

  if plot:
    fig, ax = plt.subplots(1,1, figsize=(6,6))
    ax.plot(true_pos,noise_reduction_factor)  
    ax.set_ylabel(f'Noise reduction factor. Total {len(noise_loss)} noise events')
    ax.set_xlabel('Efficiency/True Positive Rate')
    ax.set_title('Signal efficiency vs. noise reduction factor')
    ax.semilogy(True)
    ax.set_xlim(0.875,1)
    ax.grid()
    path = path + '_Signal_efficiency_vs_noise_reduction_factor.png'
    plt.savefig(path)
    
    plt.show()
    # plt.plot(false_pos, true_pos)
    # plt.show()
  return threshold_value, tpr, fpr, tnr, fnr  

def confusion_matrix(threshold_value, tpr, fpr, tnr, fnr):
  from tabulate import tabulate

  tabel = [['', 'Data-with-signals', 'Data-without-signal'],
           ['Signal detected', f'{tpr:.2f}', fpr],
           ['Noise detected', f'{fnr:.2f}', tnr]]
  print(f'Confusion matrix with threshold value at {threshold_value:.2e}')  
  print(tabulate(tabel, headers='firstrow'))  

def noise_reduction_curve_multi_models(models, path, fpr, x_test, smask_test, plot=True, x_low_lim=0.8, save_outputs=True, models_to_plot=[]):
  """
    This function takes signal and noise loss as arguments. They are 
    arrays from mse calculating.
    Bins is taken from hist
    Args: 
      models: a list of keras models
      path: where the plots saves
      fpr: False Positive Rate 
      x_low_lim: limit for lowest x value on plot (highest=1)
    Returns:
      thershold: value for a specific False Positive Ratio fpr for best model
      tpr: True positive ratio for best model
      fpr: False positive ratio for best model
      tnr: True negative ratio for best model
      fnr: False negative ratio for best model
      results[0][4]: noise reduction factor for first model

  """
  number_of_models = len(models)
  results = [0]*number_of_models
  for j in range(number_of_models):
    
    model = models[j]
    not_found_treshold_value = True
    signal_loss, noise_loss = prep_loss_values(model, x_test, smask_test)
    
    max_value = np.max(signal_loss)
    min_value = np.min(noise_loss)
    low_lim = np.floor(np.log10(min_value))
    high_lim = np.floor(np.log10(max_value))
    bins = np.logspace(low_lim,high_lim , 1000)


    threshold_value = 0
    true_pos = np.zeros(len(bins))
    false_pos = np.zeros(len(bins))
    true_neg = np.zeros(len(bins))
    false_neg = np.zeros(len(bins))
    noise_reduction_factor = np.zeros(len(bins))


    tpr = 0
    for i, limit in enumerate(bins):
    
      true_pos[i] = np.count_nonzero(signal_loss > limit)/len(signal_loss)
      false_pos[i] =np.count_nonzero(noise_loss > limit)/len(noise_loss)
      true_neg[i] = 1 - false_pos[i]
      false_neg[i] = 1 - true_pos[i]
    

      if (true_neg[i] < 1):
        noise_reduction_factor[i] = 1 / ( 1 - true_neg[i])
      else:
        noise_reduction_factor[i] = len(noise_loss)  
      
      
      if false_pos[i] < fpr and not_found_treshold_value:
        threshold_value = limit
        tpr = true_pos[i]
        not_found_treshold_value = False
        


    fnr = 1 - tpr
    tnr = 1 - fpr
    

    results[j] = [true_pos, true_neg, false_pos, false_neg, noise_reduction_factor]

    

  
  if plot:
    if len(models_to_plot) < 1:
      models_to_plot = np.arange(0,len(models))

    for k in models_to_plot:
      model_name = 'model ' + str(k+1)
      plt.plot(results[k][0],results[k][4], label=model_name)  
      
    noise_events = np.count_nonzero(~smask_test)

    plt.legend()
    plt.ylabel(f'Noise reduction factor. Total {noise_events} noise events')
    plt.xlabel('Efficiency/True Positive Rate')
    plt.title('Signal efficiency vs. noise reduction factor')
    plt.semilogy(True)
    plt.xlim(x_low_lim,1)
    plt.grid()
    if len(models) > 1:
      path = path + '/Signal_efficiency_vs_noise_reduction_factor_all_models.png'
    else:
      path = path + '_Signal_efficiency_vs_noise_reduction_factor.png'  
    plt.tight_layout()
    if save_outputs:
      plt.savefig(path)

    plt.show()

  return threshold_value, tpr, fpr, tnr, fnr, results[0][4]

def find_best_model(path,fpr, x_test, smask_test,save_output=True ):
  """
    This finction steps trough the file results.csv for finding the
    best model based on what model have the higest true positive rate 
    when false positive is equal to fpr, normaly 0.05
    Args:
      path: in what folder to search in
      fpr: false positive rate
      save_output:
      
  """
  results_path = path + '/' + 'results.csv'
  results = pd.read_csv(results_path)

  print(results)
  column = results['True pos.']
  index_of_max = column.idxmax()
  best_model = results.iloc[index_of_max]  
  

  model_path = path + '/' + best_model['Model name'] + '.h5'
  print(1)
  print(model_path)
  model = load_model(model_path)
  print(2)
  signal_loss, noise_loss = prep_loss_values(model,x_test,smask_test)
  _ = hist(path + '/' + 'best_model', signal_loss, noise_loss, plot=True)
  threshold_value, tpr, fpr, tnr, fnr, noise_reduction_factor = noise_reduction_curve_multi_models([model], path+ '/' + 'best_model',save_outputs = save_output, fpr=fpr, plot=True )
  confusion_matrix(threshold_value, tpr, fpr, tnr, fnr)
  if save_output:
    model.save((path + '/' + 'best_model'+ '.h5'))
  print(best_model['False pos.'])
  best_model['False pos.'] = fpr
  print(best_model['False pos.'])
  best_model['True pos.'] = tpr
  print(best_model)

def plot_table(path, table_name='results.csv', headers=[ 'Model name', 'Epochs', 'Batch', 'Kernel', 'Learning rate', 'Signal ratio', 'False pos.', 'True pos.', 'Latent space', 'Flops', 'Layers']):
  """
    This function plots the result from the atempt. The results are pandas dataframe.
    Args:
      path: Where the dataframe is stored
      table_name: name on the file with the result
      headers: The columns that is going to be plotted
  """
  atempt = path[-7:]
  result_path = path + '/' + table_name
  results = pd.read_csv(result_path)
  fig, ax = plt.subplots()#1,1, figsize=(12,4)
  #fig.patch.set_visible(False)


  ax.axis('off')
  ax.axis('tight')
  table = ax.table( cellText=results[headers].values,
                    colLabels=results[headers].columns,
                    loc='center'
                    )
  #table.auto_set_font_size(False)                    
  table.set_fontsize(10)                    
  table.scale(20, 1)   
  table.auto_set_column_width(col=list(range(len(results[headers].columns))))                 
  ax.set_title(f'Hyperparameters from {atempt} ', 
              fontweight ="bold")                     
  fig.tight_layout()

  savefig_path = path + '/' + atempt + '_table.png'
  plt.savefig(savefig_path,
              bbox_inches='tight',
              edgecolor=fig.get_edgecolor(),
              facecolor=fig.get_facecolor(),
              dpi=150 )
  plt.show()   
