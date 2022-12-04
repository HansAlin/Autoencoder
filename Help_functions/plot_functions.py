from cmath import isnan
from ctypes import sizeof
from turtle import title
#from lib2to3.pytree import _Results
from tensorflow import keras
import numpy as np
import os
from matplotlib import pyplot as plt
import pandas as pd
from tensorflow.keras.models import load_model
import Help_functions.data_manage as dm
from scipy import integrate
from Model_classes.NewPhysicsAutoencoder import NewPhysicsAutoencoder
from Model_classes.SecondCNNModel import SecondCNNModel
from Model_classes.DenseModel import DenseModel
from Model_classes.ConvAutoencoder import ConvAutoencoder
import Help_functions.creating_models as cm
from keras_flops import get_flops


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

def plot_signal_noise(x,smask):
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

def plot_signal_noise(plot_area=False, save_path='/home/halin/Autoencoder/Pictures/Signal_noise_examples.png'):
  test_plot = np.load('/home/halin/Autoencoder/Data/plot_examples.npy')
  x = np.linspace(0,100,100)
  
  fig, (ax1,ax2) =plt.subplots(1,2,figsize=(8,4))
  if plot_area:
    fig, ax1 =plt.subplots(1,1,figsize=(4,4))
   
    
  ax1.plot(test_plot[:,0])
  ax1.set_title("Signal")
  ax1.set_xlabel('ns')
  ax1.set_ylabel('Voltage')
  if plot_area:
    ax1.fill_between(
          x=x, 
          y1= test_plot[:,0], 
          where= (0 < x)&(x < 100),
          color= "b",
          alpha= 0.2)
    plt.annotate("Area", xy=(59,4), xytext=(70,6),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))      
  ax1.grid()
  
  if  not plot_area:
    ax2.plot(test_plot[:,3])
    
    ax2.set_title("Noise")
    ax2.set_xlabel('ns')
    ax2.grid()
  #ax2.set_ylabel('Voltage')
  #fig.suptitle("Data examples")
  fig.legend()
  fig.tight_layout()
  fig.savefig(save_path)
  


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
  plt.cla()

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

def alternative_loss_values(model, x, smask):
  """
    This function predict values using keras.predict and calculates mse
    and finds the greatest value in the bin.
    Args:
      model: keras model
      x: the test data shape (Any, 100, 1)
      smask: smask for x 
    Returns: 
      max_signal_diff, max_noise_diff  
  """
  x_noise = x[~smask]
  x_pred_noise = model.predict(x_noise)
  x_signal = x[smask]
  x_pred_signal = model.predict(x_signal)

  max_noise_diff = np.max((x_noise - x_pred_noise)**2, axis=1)
  max_signal_diff = np.max( (x_signal - x_pred_signal)**2,axis=1)

  return max_signal_diff, max_noise_diff

def integrated_loss_values(model, x, smask):
  data_bins = np.size(x[0])
  x_signal = x[smask].reshape((121594,100))
  x_noise = x[~smask].reshape((243188,100))
  
  x_pred_noise = model.predict(x_noise).reshape((243188,100))
  x_pred_signal = model.predict(x_signal).reshape((121594,100))

  noise_residual = np.abs(x_noise-x_pred_noise)
  signal_residual = np.abs(x_signal-x_pred_signal)
  len_signal = len(signal_residual)
  len_noise = len(noise_residual)
  signal_residual_integration_value = np.zeros(len_signal)
  noise_residual_integration_value = np.zeros(len_noise) 
  time_range = np.linspace(0,0.1,100)
  for i in range(len_signal):
    signal_residual_integration_value[i] = integrate.simps(y=signal_residual[i], x=time_range)
  for i in range(len_noise):  
    noise_residual_integration_value[i] = integrate.simps(y=noise_residual[i], x=time_range)  

  return signal_residual_integration_value, noise_residual_integration_value

def costum_loss_values(model, x, smask):
  
  '''
  This function performes a mean square error 
  calculation based on the values that the model have 
  predicted.

  Arguments:
    model: a Keras Autencoder model with same input size as output size.
    x: The data
    smask: is where the signals and noise are in the data.

  Returns:
    signal_loss: the mse value for the signal
    noise_loss: the mse for the noise  

    signal_loss, noise_loss
  '''
  x_noise = x[~smask]
  x_pred_noise = model.predict(x_noise)
  x_signal = x[smask]
  x_pred_signal = model.predict(x_signal)
  len_noise = len(x_noise[0])
  len_signal = len(x_signal[0])
  noise_residue = x_noise - x_pred_noise
  signal_residue = x_signal - x_pred_signal
  
  noise_loss = np.sum((noise_residue)**2, axis=1)/len_noise
  signal_loss = np.sum( (signal_residue)**2,axis=1)/len_signal

  return signal_loss, noise_loss

def costum_loss_values_2(model , x, smask):
  x_noise = x[~smask]
  x_pred_noise = model.predict(x_noise)
  x_signal = x[smask]
  x_pred_signal = model.predict(x_signal)
  len_noise = len(x_noise[0])
  len_signal = len(x_signal[0])
  noise_residue = x_noise - x_pred_noise
  signal_residue = x_signal - x_pred_signal
  abs_noise_residue = np.abs(noise_residue)
  abs_signal_residue = np.abs(signal_residue)
  abs_max_noise_residue = np.max(abs_noise_residue, axis=1)
  abs_max_signal_residue = np.max(abs_signal_residue, axis=1)

  return abs_max_signal_residue, abs_max_noise_residue

def costum_loss_values_3(model , x, smask):
  x_noise = x[~smask]
  x_pred_noise = model.predict(x_noise)
  x_signal = x[smask]
  x_pred_signal = model.predict(x_signal)
  len_noise = len(x_noise[0])
  len_signal = len(x_signal[0])
  noise_residue = x_noise - x_pred_noise
  signal_residue = x_signal - x_pred_signal
  # noise_residue = noise_residue
  # signal_residue = signal_residue
  
  signal_integration_value = np.zeros(len(signal_residue))
  noise_integration_value = np.zeros(len(noise_residue)) 
  time_range = np.linspace(0,0.1,100)

  for i in range(len(signal_residue)):
    signal_integration_value[i] = integrate_graph(signal_residue[i],time_range)
  for i in range(len(noise_residue)):  
    noise_integration_value[i] = integrate_graph(noise_residue[i],time_range)



  return signal_integration_value, noise_integration_value # Loss values

def costum_loss_values_4(x, smask, save_path='/home/halin/Autoencoder/Pictures'):
  x_noise = x[~smask]
  x_signal = x[smask]
  noise_residue = x_noise
  signal_residue = x_signal 
  # noise_residue = noise_residue
  # signal_residue = signal_residue
  
  signal_integration_value = np.zeros(len(signal_residue))
  noise_integration_value = np.zeros(len(noise_residue)) 
  time_range = np.linspace(0,0.1,100)

  for i in range(len(signal_residue)):
    signal_integration_value[i] = integrate_graph(signal_residue[i],time_range)
  
  for i in range(len(noise_residue)):  
    noise_integration_value[i] = integrate_graph(noise_residue[i],time_range)


  return signal_integration_value, noise_integration_value # Loss values


def integrate_graph(y , x):
  pos = 0
  while pos < len(y) - 1:
    if np.abs(y[pos + 1]) != 0 and np.abs(y[pos]) != 0:
      if y[pos + 1]/np.abs(y[pos + 1]) != y[pos]/np.abs(y[pos]):
        k = (y[pos + 1] - y[pos])/(x[pos + 1] - x[pos])
        delta_x = -y[pos]/k
        y = np.insert(y, pos+1,0)
        x_pos = x[pos] + delta_x
        x = np.insert(x, pos+1, x_pos)
    pos += 1
  if len(y) == 100 or len(x) == 100:
    print()
    y = y.reshape((len(y),))
    x = x.reshape((len(x),))

  
  integrated_value = integrate.simps(y=np.abs(y), x=x)
  return integrated_value

def max_loss_values(model, x, smask):
  x_signal = x[smask].reshape((121594,100))
  x_noise = x[~smask].reshape((243188,100))
  
  x_pred_noise = model.predict(x_noise).reshape((243188,100))
  x_pred_signal = model.predict(x_signal).reshape((121594,100))

  noise_residual = np.abs(x_noise-x_pred_noise)
  signal_residual = np.abs(x_signal-x_pred_signal) 

  len_signal = len(signal_residual)
  len_noise = len(noise_residual) 

  signal_value = np.zeros(len_signal)
  noise_value = np.zeros(len_noise) 

  signal_value = np.max(signal_residual, axis=1)
  noise_value = np.max(noise_residual,axis=1)

  return signal_value, noise_value

def hist(path, signal_loss, noise_loss, resolution=100, plot=True):
  
  max_value = np.max(signal_loss)
  min_value = np.min(noise_loss)
  low_lim = np.floor(np.log10(min_value))
  high_lim = np.floor(np.log10(max_value))
  bins = np.logspace(low_lim,high_lim , resolution)
  plt.close('all') 
  weights = np.ones(len(noise_loss))*0.5
  if plot:
    
    ax1 = plt.hist(noise_loss, bins=bins, label='Noise', alpha=0.5, weights=weights) #, log=True,  histtype='step') # )
   
    ax2 = plt.hist(signal_loss, bins=bins,  label='Signal',alpha=0.5)#, log=True,  histtype='step', density=True, label='Signal') #)
    plt.xscale('log')

    
    #plt.yscale('log')
    plt.xlim(right=1000, left=0.1)
    ## Explainary bars 
    # plt.plot([0.2,0.2],[0,30000], color='black', linestyle='dashed')
    # plt.annotate("Threshold", xy=(0.2,100), xytext=(0.001, 100),
    #         arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
    # plt.annotate("", xy=(0.2/10,1000), xytext=(0.2*10, 1000),
    #         arrowprops=dict(arrowstyle="<->", connectionstyle="arc3"))
    plt.xlabel('Mean squared error')
    plt.ylabel('Counts')
    plt.legend()
    path = path + '_hist.png'
    plt.grid()
    plt.savefig(path)
    plt.cla()
    

    
    
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
    plt.cla()

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
    plt.cla()
    
  return threshold_value, tpr, fpr, tnr, fnr  

def confusion_matrix(threshold_value, tpr, fpr, tnr, fnr):
  from tabulate import tabulate

  tabel = [['', 'Data-with-signals', 'Data-without-signal'],
           ['Signal detected', f'{tpr:.2f}', fpr],
           ['Noise detected', f'{fnr:.2f}', tnr]]
  print(f'Confusion matrix with threshold value at {threshold_value:.2e}')  
  print(tabulate(tabel, headers='firstrow'))  

def noise_reduction_curve_single_model(model_name, save_path, fpr,signal_loss, noise_loss, plot=True, x_low_lim=0.8):
  """
    This function takes signal and noise loss as arguments. They are 
    arrays from mse calculating. It calculates tpr, fpr, noise_reduction_factor
    tnr, se below.
    
    Args: 
      model_name: model_name
      save_path: where the plots saves incl. name e.g. your_path/model_name
      fpr: False Positive Rate 
      x_low_lim: limit for lowest x value on plot (highest=1)
      signal_loss: calculated in prep_loss_values()
      noise_loss: calculated in prep_loss_values()
    Returns:
      thershold: value for a specific False Positive Ratio fpr for best model
      tpr: True positive ratio for best model
      fpr: False positive ratio for best model
      tnr: True negative ratio for best model
      fnr: False negative ratio for best model
      noise_reduction_factor: noise reduction factor for first model
      true_pos: true positive rate

  """
  #TODO signal_loss and noise_loss comes from one model but several models can be loaded to
  # the function.!!!!
  
  results = []

  not_found_treshold_value = True
  
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
 
  if plot:
    if model_name[0] == '_':
      model_name = 'M' + model_name

    plt.plot(true_pos,noise_reduction_factor, label=model_name)  
      
    noise_events = len(noise_loss)

    plt.legend()
    plt.ylabel(f'Noise reduction factor. Total {noise_events} noise events')
    plt.xlabel('Efficiency/True Positive Rate')
    plt.title('Signal efficiency vs. noise reduction factor')
    plt.semilogy(True)
    plt.xlim(x_low_lim,1)
    plt.grid()
    #TODO fix save path
    save_path = save_path + '_Signal_efficiency_vs_noise_reduction_factor.png'  
    plt.tight_layout()
    
    plt.savefig(save_path)

    plt.show()
    plt.cla()

  return threshold_value, tpr, fpr, tnr, fnr, noise_reduction_factor, true_pos

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

def plot_table(results, save_path, headers=['Model name',
																'Model type',
                                'Batch', 
                                'Kernel', 
                                #'Learning rate', 
                                'Latent space', 
                                'Number of filters', 
																#'Conv. in row',
                                'Flops',
                                #'True pos. array',
                                #'Act. last layer',
                                #'Activation func. rest',
							                  ], sufix='', prefix=''):
  """
    This function plots the result from the atempt. The results are pandas dataframe.
    Args:
      path: Where the dataframe is stored
      table_name: name on the file with the result
      headers: The columns that is going to be plotted
  """
  atempt = save_path[-4:-1]
  fig, ax = plt.subplots()#1,1, figsize=(12,4)
  # #fig.patch.set_visible(False)
  if prefix == '':
    prefix = save_path[-8:-4]
  ax.axis('off')
  ax.axis('tight')
  table = ax.table( cellText=results[headers].values,
                    colLabels=results[headers].columns,
                    loc='center'
                    )
  #table.auto_set_font_size(False)                    
  table.set_fontsize(8)                    
  #table.scale(20, 1)   
  table.auto_set_column_width(col=list(range(len(results[headers].columns))))                 
  ax.set_title(f'Hyperparameters from {atempt} ', 
              fontweight ="bold")                     
  fig.tight_layout()

  savefig_path = save_path + prefix + atempt + sufix + '_table.png'
  plt.savefig(savefig_path,
              
              bbox_inches='tight',
              edgecolor=fig.get_edgecolor(),
              facecolor=fig.get_facecolor(),
              dpi=150
               )
  #df_styled = results.style.background_gradient()             
  plt.show() 
  plt.cla()  

def noise_reduction_from_results(results, best_model, x_low_lim=0.8, save_path='', name_prefix=''):
  """
    This function takes results from a pandas dataframe
    and plots the different noise reduction curves
    Args: 
      results: a pandas dataframe
      save_outputs: saving the plot or not
      x_low_lim: limit for lowest x value on plot (highest=1)

  """
  plt.close('all') 
  fig, ax = plt.subplots()  
  if isinstance(best_model, pd.DataFrame):
    value1 = best_model['True pos. array'][0]
    value2 = best_model['Noise reduction'][0]
    tpr = convert_result_dataframe_string_to_list(value1) 
    nr = convert_result_dataframe_string_to_list(value2) 
    model_name = best_model['Model name'][0]
    ax.plot(tpr, nr, label='Best model ' + model_name)
  linestyles = ['-', '--', '-.', ':']  
  i = 0
  nr = [0]
  for index, row in results.iterrows():
    tpr = convert_result_dataframe_string_to_list(row['True pos. array'])
    nr = convert_result_dataframe_string_to_list(row['Noise reduction'])
    name = row['Model name']
    if name[0] == '_':
      name = 'M' + name
    if i > 3:
      i = i - 4  
    linestyle = linestyles[i]

    ax.plot(tpr, nr, label=name, linestyle=linestyle) 
    i += 1
      
  
  total_noise_events = int(np.max(nr))  
  ax.legend()
  y_label = f'Noise reduction factor. Total {total_noise_events} noise events.'
  plt.ylabel(y_label)
  plt.xlabel('Efficiency/True Positive Rate')
  plt.title('Signal efficiency vs. noise reduction factor')
  plt.semilogy(True)
  plt.xlim(x_low_lim,1)
  plt.grid()
  plt.tight_layout()
 
  if save_path != '':
    save_path = save_path + name_prefix + 'Signal_efficiency_vs_noise_reduction_factor.png' 
    plt.savefig(save_path)

  plt.show()
  plt.cla()

def convert_result_dataframe_string_to_list(result_string):
  
  result_string = result_string[1:]
  result_string = result_string[:-1]
  result_string = result_string.replace('\n', ' ')
  result_string = result_string.split(' ')
  temp_tpr = []
  for x in result_string:
    if x != '':
      temp_tpr.append(float(x))
  return temp_tpr

def convert_array_to_string(array_list):
  result_string = '['
  length = len(array_list)
  for i in range(length):
    if i != (length - 1):
      result_string += str(array_list[i]) + ' '
    else:
      result_string += str(array_list[i])
  result_string += ']'
  return result_string  

def plot_performance(model, x_test, smask_test, save_path, std, mean):
  
  #model = load_model(path)
  #print(model.summary())

  
  x_noise = x_test[~smask_test]
  x_noise = x_noise[:10]      #small subset
  x_pred_noise = model.predict(x_noise)[0]
  x_noise = x_noise[0]
  x_noise = dm.unnormalizing_data(x_noise, std=std, mean=mean)
  x_pred_noise = dm.unnormalizing_data(x_pred_noise, std=std, mean=mean)
  res_noise = x_noise - x_pred_noise

  x_signal = x_test[smask_test]
  x_signal = x_signal[:10]    #subset
  x_pred_signal = model.predict(x_signal)[0]
  x_signal = x_signal[0]
  x_signal = dm.unnormalizing_data(x_signal, std=std, mean=mean)
  x_pred_signal = dm.unnormalizing_data(x_pred_signal, std=std, mean=mean)
  res_signal = x_signal - x_pred_signal
  end_name = save_path[-15:]
  if end_name[0] == 'N':
    end_name = 'C' + end_name
  fig, axis = plt.subplots(2,2, figsize=(15,15))

  axis[0,0].plot(x_noise, label="Original noise")
  axis[0,0].plot(x_pred_noise, label="Predicted noise")
  axis[0,0].legend()
  axis[0,0].grid()
  axis[0,0].set_title(f'Noise')

  axis[0,1].plot(x_signal, label="Original signal")
  axis[0,1].plot(x_pred_signal, label="Predicted signal")
  axis[0,1].set_title(f'Signal')
  axis[0,1].grid()
  axis[0,1].legend()

  axis[1,0].plot(res_noise, label='Residual')
  axis[1,0].set_title(f'Noise residuals')
  axis[1,0].grid()
  axis[1,0].legend()

  axis[1,1].plot(res_signal, label='Residual')
  axis[1,1].set_title(f'Signal residuals')
  axis[1,1].grid()
  axis[1,1].legend()
  fig.suptitle(f'Model {end_name}', fontsize=12)
  
  plt.tight_layout()
  plt.savefig(save_path + '_Signal_and_noise_pred \n')
  plt.show()
  plt.cla()

def plot_single_performance(model, plot_examples, save_path, std, mean, prefix_header='', sufix='', plot_title=''):
  plot_examples = plot_examples.reshape((2,100,1))
  smask=np.asarray([True,False])
  x_pred_example = model.predict(plot_examples)
  signal_loss, noise_loss = costum_loss_values(model=model, x=plot_examples, smask=smask)
  print('Costum 1 loss values')
  print(noise_loss)
  print(signal_loss)
  print('Costum 2 loss values')
  signal_loss, noise_loss =costum_loss_values_2(model=model,x=plot_examples,smask=smask)
  print(noise_loss)
  print(signal_loss)
  print('Costum 3 loss values')
  signal_loss, noise_loss = costum_loss_values_3(model=model, x=plot_examples, smask=smask)
  print(noise_loss)
  print(signal_loss)

  x_example = dm.unnormalizing_data(plot_examples, std=std, mean=mean)
  x_pred_example = dm.unnormalizing_data(x_pred_example, std=std, mean=mean)
  res_example = x_example - x_pred_example

  end_name = save_path[-15:]
  if end_name[0] == 'N':
    end_name = 'C' + end_name
  plt.close('all')  
  fig, axis = plt.subplots(2,2, figsize=(8,8))

  axis[0,0].plot(x_example[0], label="Original signal")
  axis[0,0].plot(x_pred_example[0], label="Predicted signal")
  axis[0,0].legend()
  axis[0,0].grid()
  axis[0,0].set_title(f'Signal')
  axis[0,0].set_ylabel('Voltage')
  

  axis[0,1].plot(x_example[1], label="Original noise")
  axis[0,1].plot(x_pred_example[1], label="Predicted noise")
  axis[0,1].set_title(f'Noise')
  axis[0,1].grid()
  axis[0,1].legend()
  

  axis[1,0].plot(res_example[0], label='Residual')
  axis[1,0].set_title(f'Signal residuals')
  axis[1,0].grid()
  axis[1,0].legend()
  axis[1,0].set_ylabel('Voltage')
  axis[1,0].set_xlabel('ns')

  axis[1,1].plot(res_example[1], label='Residual')
  axis[1,1].set_title(f'Noise residuals')
  axis[1,1].grid()
  axis[1,1].legend()
  axis[1,1].set_xlabel('ns')
  plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])
  #fig.text(y=2,s='Voltage')
  if plot_title != '':
    fig.suptitle(plot_title, fontsize=12)
    plt.savefig(save_path + plot_title + f'_Signal_and_noise_pred_ex_{sufix}')
  else:  
    fig.suptitle(f'{prefix_header} Model {end_name} ex. {sufix} \n', fontsize=12)
    plt.savefig(save_path + f'_Signal_and_noise_pred_ex_{sufix}')
  
  
  plt.show()
  plt.cla()

def integration_test(x_test, smask_test, save_path, plot=False ):

    #test_size = 100000
    signal = x_test[smask_test].reshape((121594,100))#[:test_size]
    noise = x_test[~smask_test].reshape((243188,100))#[:test_size]
    signal = np.abs(signal)
    noise = np.abs(noise)
    signal_integration_value = np.zeros(len(signal))
    noise_integration_value = np.zeros(len(noise)) 
    time_range = np.linspace(0,0.1,100)

    for i in range(len(signal)):
      signal_integration_value[i] = integrate.simps(y=signal[i], x=time_range)
    for i in range(len(noise)):  
      noise_integration_value[i] = integrate.simps(y=noise[i], x=time_range)

    max_value = np.max(signal_integration_value)
    min_value = np.min(noise_integration_value)
    low_lim = np.floor(np.log10(min_value))
    high_lim = np.floor(np.log10(max_value))
    bins = np.logspace(low_lim, high_lim, 1000)

    if plot:
      _ = plt.hist(noise_integration_value, alpha=0.5, bins=bins, log=True, density=True)
      _ = plt.hist(signal_integration_value, alpha=0.5, bins=bins, log=True, density=True)
      plt.xscale('log')
      plt.xlim([min_value,max_value])
      plt.show()
      plt.savefig(save_path + 'integration_test_hist.png')
      plt.cla()
      
    true_pos = np.zeros(len(bins))
    false_pos = np.zeros(len(bins))


    true_neg = np.zeros(len(bins))
    false_neg = np.zeros(len(bins))
    noise_reduction_factor = np.zeros(len(bins))
    for i, limit in enumerate(bins):
      true_pos[i] = np.count_nonzero(signal_integration_value > limit)/len(signal_integration_value)
      false_pos[i] = np.count_nonzero(noise_integration_value > limit)/len(noise_integration_value)
      true_neg[i] = 1 - false_pos[i]
      false_pos[i] = 1 - true_pos[i]
      if (true_neg[i] < 1):
          noise_reduction_factor[i] = 1 / ( 1 - true_neg[i])
      else:
          noise_reduction_factor[i] = len(noise_integration_value)  
    true_neg = 1 - false_pos
    false_pos = 1 - true_pos
    if plot:
      plt.plot(true_pos,noise_reduction_factor, label="Base line")  
            
      noise_events = np.count_nonzero(~smask_test)

      plt.legend()
      plt.ylabel(f'Noise reduction factor. Total {noise_events} noise events')
      plt.xlabel('Efficiency/True Positive Rate')
      plt.title('Signal efficiency vs. noise reduction factor')
      plt.semilogy(True)
      plt.xlim(0.8,1)
      plt.grid()
      plt.show()  
      plt.savefig(save_path+ 'integration_test_plot.png')
      plt.cla()

    return true_pos, noise_reduction_factor 
def from_string_to_numpy(column):
  
  if isinstance(column, (np.ndarray, pd.DataFrame)):
    column_array = np.asarray(column)[0][0]
  else:
    column_array = column  
  column_array = column_array.split('[')[1]
  column_array = column_array.split(']')[0]
  column_array = column_array.split(' ')
  #column_array = column_array.split('\n')
  column_array = [float(x) for x in column_array]
  column_array = np.asarray(column_array)
  return column_array
  
def change_new_results(results, folder, x_test,smask_test, prefix='', folder_path = '/home/halin/Autoencoder/Models/'):
  """
    This function calculates new loss values using a costum loss function
    and adds the new values to the dataframe and saves to csv
    Plots a histogram and a noise reduction curve
    Args:
      results: dataframe with earlier data
      folder: folder where dataframe is stored
      x_test: test data
      smask_test: smask of data
      prefix: prefix if any
      folder_path: the folder where alle the models are stored
   
  """
  (rows, cols) = results.shape
  
  for model_number in range(1,rows + 1):
      model_path = folder_path + f'CNN_{folder}/CNN_{folder}_model_{model_number}.h5'
      save_path = folder_path + f'CNN_{folder}/' + prefix + f'CNN_{folder}_model_{model_number}'
      try:
       model = load_model(model_path) 
      except OSError as e:
        print(f'No model {model_number}')
        continue
      
      signal_loss, noise_loss = costum_loss_values_3(model,x_test,smask_test)
      #_ = hist(path=save_path,signal_loss=signal_loss, noise_loss=noise_loss)
      model_name_string = f'Test_CNN_{folder}_model_{model_number}'
      threshold_value, tpr, fpr, tnr, fnr, noise_reduction_factor, true_pos = noise_reduction_curve_single_model(model_name=model_name_string, save_path=save_path, signal_loss=signal_loss, noise_loss=noise_loss, fpr=0.05, x_low_lim=0.95,plot=False)
      # Replace old values with new
      noise_reduction_factor = convert_array_to_string( noise_reduction_factor)
      true_pos = convert_array_to_string(true_pos)
      results.loc[[model_number - 1], ['Noise reduction']] = noise_reduction_factor #replace({'Noise reduction': {model_number - 1 : noise_reduction_factor}})
      results.loc[[model_number - 1], ['True pos. array']] = true_pos
      results.loc[[model_number - 1], ['True pos.']] = tpr
      results.loc[[model_number - 1], ['Threshold value']] = threshold_value
  result_csv = folder_path + f'CNN_{folder}/' + prefix + 'results.csv'
  results.to_csv(result_csv) 
  return result_csv

def find_best_model_in_folder(headers,
                 start_model=101, 
                 end_model=150, 
                 folder_path='/home/halin/Autoencoder/Models/', 
                 save_path='/home/halin/Autoencoder/Models/',
                 folder='mixed_models/',
                 save_name='', 
                 number_of_models=10, 
                 terms_of_condition ='',
                 value_of_condition ='', 
                 comparison = 'equal', 
                 prefix='mixed_models', 
                 x_low_lim=0.9, 
                 result_path = '',
                  ):
  preprefix = 'sorted_'               
  single_dataframe = False                
  if result_path != '':
    single_dataframe = True
  max_value = [0]*number_of_models
  import re
  name_best_model = ['']*number_of_models
  result_dic = {}
  if result_path != '':
    start_model = 1
    end_model = 2

  for i in range(start_model,end_model):
    
    if not single_dataframe:
      result_path = folder_path + f'CNN_{i}/' + prefix + 'results.csv'
    else:
      result_path = result_path
    try:
      results = pd.read_csv(result_path)
    except OSError as e:
      print(f'No file in folder {result_path}')
      continue
    
    in_data = False
    if terms_of_condition == '':
      in_data = True
    else:
      for col in results.columns:
        if col == terms_of_condition:
          # String comparision
          if comparison == 'equal':
            results = results[results[terms_of_condition] == value_of_condition]
            in_data = True
          if comparison == 'greater':
            results = results[results[terms_of_condition] >= value_of_condition]
            in_data = True
          if comparison == 'less':  
            results = results[results[terms_of_condition] <= value_of_condition]
            in_data = True
           

    if (in_data): # or (terms_of_condition == ''):
      (rows, cols) = results.shape
      results = results.reset_index()
      for j in range(0,rows):
        # value = results.at([j],[terms_of_condition])
        # if value == value_of_condition or value_of_condition == '':
          noise_reduction = results.loc[[j],['Noise reduction']]
          noise_reduction = from_string_to_numpy(noise_reduction)
          noise_reduction = np.flip(noise_reduction)
          x_integration = results.loc[[j],['True pos. array']]
          x_integration = from_string_to_numpy(x_integration)
          x_integration = np.flip(x_integration)
          model_name = results.loc[[j], ['Model name']].squeeze()
          
          
          integration_value = np.trapz(noise_reduction,x=x_integration)
          #integration_value = np.sum(noise_reduction)
          for k in range(len(max_value)):
            if integration_value > max_value[k]:
                max_value.insert(k, integration_value)
                max_value.pop()
                name_best_model.insert(k, model_name)
                name_best_model.pop()
                break
  # length =  len(name_best_model)           
  # for i in range(length):
  #   if name_best_model[i] == '':
  #     name_best_model.pop(i)
  #     length -= 1
  #     i -= 1
  
  best_models = list(filter(lambda a: a != '',name_best_model))

  path=folder_path
  save_result_path = save_path + prefix + preprefix + 'results.csv'

  if len(best_models) != 0:
    print(f"Number of models found: {len(best_models)}")
    save_result_path = dm.make_dataframe_of_collections_of_models(best_models=best_models,save_path=save_result_path,path=folder_path, prefix=prefix)
    results = pd.read_csv(save_result_path)
    plot_table(results,
                save_path=save_path, 
                headers=headers,
                prefix=prefix)
    
    noise_reduction_from_results(results=results, best_model='',x_low_lim=x_low_lim, save_path=save_path , name_prefix=prefix)   
  else:
    print("No models found!")
  return save_result_path

def find_weird_signals_noise(signal_loss, noise_loss, signal, loss, limit):
  weird_signal_index = signal_loss < limit
  weird_noise_index = noise_loss < limit
  signal_index_array = np.where(weird_signal_index)
  noise_index_array = np.where(weird_noise_index)

  number_of_weird_signals = len(signal_index_array)
  number_of_weird_noise = len(noise_index_array)
  for index in signal_index_array[0]:
    single_signal = signal[index]
    time = range(0,100)
    plt.plot(time, single_signal)
    plt.savefig('/home/halin/Autoencoder/Models/plots/wierd.png')

  return 0  

def find_values_from_model(folder, model_number, values_of_interest = ['Number of filters', 'Latent space']):
  result_path = f'/home/halin/Autoencoder/Models/CNN_{folder}/results.csv'
  try:
    results = pd.read_csv(result_path)
  except OSError as e:
    print(f'No file in folder {result_path}')
    return 0 

  return_values = [0]*len(values_of_interest)
  for i in range(0,len(values_of_interest)):
    col = values_of_interest[i]
    model_name = f'CNN_{folder}_model_{model_number}'
    try:
      value = results[results['Model name'] == model_name][col].values[0]
    except KeyError:
      print("No value from model")
      continue 
    
    if isinstance(value, str):
      string_value = value
      string_value = string_value.replace('[','')
      string_value = string_value.replace(']','')

      array_value = np.fromstring(string_value, dtype=int, sep=',').tolist()
      if len(array_value) == 0:
        return_values[i] = value
      else: 
        return_values[i] = array_value  
    else:
      return_values[i] = value
  return return_values  

def create_models_from_data_frame(result_path, path,plot=True):
  all_samples = False
  test_run = True
  x_low_lim = 0.8
  epochs = 2
  fpr = 0.05
  plot = True

  try:
    results = pd.read_csv(result_path)
  except OSError as e:
    print(f'No file in  {result_path}')
  new_results = pd.DataFrame(columns=['Model name',
																'Model type',
                                'Epochs',
                                'Batch', 
                                'Kernel', 
                                'Learning rate', 
                                'False pos.', 
                                'True pos.', 
                                'Threshold value', 
                                'Latent space', 
                                'Number of filters', 
																'Conv. in row',
                                'Flops',
                                'Layers', 
                                'Noise reduction',
                                'True pos. array',
                                'Signal loss',
                                'Noise loss',
                                'Act. last layer',
                                'Activation func. rest',
																'Signal ratio'])
  data_url = '/home/halin/Autoencoder/Data/'
  plot_examples = np.load('/home/halin/Autoencoder/Data/plot_examples.npy')
  
  x_test, y_test, smask_test, signal, noise, std, mean = dm.load_data(all_signals_for_testing=True,
																																		 all_samples=all_samples,						
																																		 data_path=data_url, 
																																		 small_test_set=1000,
																																		 number_of_files=10)
  (rows, cols) = results.shape
  for row in range(rows):
    model_name = results.loc[[row], ['Model name']].values[0][0]
    folder = model_name[4:7]
    model_number = model_name[14:]
    [model_name,
    model_type, 
    batch,
    kernel,
    learning_rate,
    latent_size,
    filters,
    conv_in_row,
    last_activation_function,
    activation_function,
     ]  = find_values_from_model(folder=folder,
                            model_number=model_number,
                            values_of_interest=['Model name',
																'Model type',
                                'Batch', 
                                'Kernel', 
                                'Learning rate', 
                                'Latent space', 
                                'Number of filters', 
																'Conv. in row',
                                'Act. last layer',
                                'Activation func. rest'])
    if isnan(conv_in_row) or conv_in_row == 0:
      conv_in_row = 1
    if   not isinstance(activation_function, str):
      activation_function = ''
    if not isinstance(last_activation_function,str):  
      last_activation_function = 'lienear'  
    kernel = [kernel]  
    if model_type == 'ConvAutoencoder' or model_type == '':
      (encoder, decoder, autoencoder) = ConvAutoencoder.build(data=x_test,
                                      filters=filters, 
                                      activation_function=activation_function,
                                      latent_size=latent_size,
                                      kernel=kernel,
                                      last_activation_function=last_activation_function,
                                      convs=conv_in_row )
    elif model_type == 'NewPhysicsAutoencoder':
      (encoder, decoder, autoencoder) = NewPhysicsAutoencoder.build(data=x_test,
                                      filters=filters, 
                                      activation_function=activation_function,
                                      latent_size=latent_size,
                                      kernel=kernel,
                                      last_activation_function=last_activation_function )
    elif model_type == 'SecondCNNModel':
      (encoder, decoder, autoencoder) = SecondCNNModel.build(data=x_test,
                                      filters=filters, 
                                      activation_function=activation_function,
                                      latent_size=latent_size,
                                      kernel=kernel,
                                      last_activation_function=last_activation_function,
                                      convs=conv_in_row )
    elif model_type == 'DenseModel':
      (encoder, decoder, autoencoder) = DenseModel.build(data=x_test,
                                      filters=filters, 
                                      activation_function=activation_function,
                                      latent_size=latent_size,
                                      kernel=kernel,
                                      last_activation_function=last_activation_function )

    adam = keras.optimizers.Adam(learning_rate=learning_rate) 
    autoencoder.compile(
          loss = 'mse',
          optimizer = adam,
          metrics = ['mse','mae','mape'] )
    print(autoencoder.summary())  
    x_train, smask_train, y_train = dm.create_data(signal=signal, 
                                                  noise=noise, 
                                                  test_run=test_run, 
                                                  signal_ratio=0,
                                                  maximum_ratio=0)      
    trained_autoencoder = cm.train_autoencoder(model=autoencoder,
                                                x_train=x_train,
                                                  epochs=epochs,
                                                  batch=batch,
                                                    verbose=1)
    flops = get_flops(autoencoder)
    save_path = path + model_name
    autoencoder.save((save_path  + '.h5'))
    if plot:
      loss_plot(save_path, trained_autoencoder)
      sufix = 1
      to_plot = np.vstack((plot_examples[:,0], plot_examples[:,2]))
      plot_single_performance(autoencoder,to_plot,save_path,std,mean, sufix=sufix)
      plt.cla()
      sufix = 2
      to_plot = np.vstack((plot_examples[:,1], plot_examples[:,3]))
      plot_single_performance(autoencoder,to_plot,save_path,std,mean, sufix=sufix)
      plt.cla()
    signal_loss, noise_loss = costum_loss_values(autoencoder,x_test,smask_test)
    bins = hist(save_path, signal_loss, noise_loss, plot=plot)
    threshold_value, tpr, fpr, tnr, fnr, noise_reduction_factors, true_pos_array = noise_reduction_curve_single_model(
                                                                  model_name=model_name,
                                                                  save_path=save_path,
                                                                  fpr=fpr, 
                                                                  plot=plot, 
                                                                  signal_loss=signal_loss, 
                                                                  noise_loss=noise_loss,
                                                                  x_low_lim=x_low_lim)                                                                   
    results.loc[row, 'False pos.'] = fpr
    results.loc[row, 'True pos.'] = tpr
    results.loc[row, 'Threshold value'] = threshold_value
    results.loc[row, 'Flops'] = flops
    results.loc[row,'Noise reduction'] = convert_array_to_string(noise_reduction_factors)
    results.loc[row, 'True pos. array'] = convert_array_to_string(true_pos_array)
    results.loc[row, 'Signal loss'] = convert_array_to_string(signal_loss)
    results.loc[row, 'Noise loss'] = convert_array_to_string(noise_loss)
    results.loc[row, 'Epochs'] = epochs
  results.to_csv(path+'results.csv')
  plot_table(results=results, save_path=path,headers=['Model name',
																'Model type',
                                #'Learning rate', 
                                'Latent space', 
                                'Number of filters', 
																#'Conv. in row',
                                'Flops',
                                #'True pos. array',
                                #'Act. last layer',
                                #'Activation func. rest',
							                  ])
  noise_reduction_from_results(results=results,
                                  best_model='',
                                  x_low_lim=x_low_lim,
                                  save_path=path,
                                  name_prefix='')

def plot_several_NR_curves(model, labels, prefix, save_path, x_low_lim, label_title):
  number_of_models = len(model)
  plt.close('all') 
  fig, ax = plt.subplots()
  linestyles = ['-', '--', '-.', ':'] 
  nr = [0]
  i = 0
  for j in range(number_of_models):
      folder, model_number = model[j]
      model_path = f'/home/halin/Autoencoder/Models/CNN_{folder}/'
      model_path = model_path + prefix + 'results.csv'
      try: 
          results = pd.read_csv(model_path)
      except OSError as e:
        print(f'No file in folder {model_path}')
        continue    
      model_name = f'CNN_{folder}_model_{model_number}'
      try:
        nr = results[results['Model name'] == model_name]['Noise reduction'].values[0]
      except KeyError:
        print("No value from model")
        continue
      try:
          tpr = results[results['Model name'] == model_name]['True pos. array'].values[0]
      except KeyError:
        print("No value from model")
        continue
      tpr = convert_result_dataframe_string_to_list(tpr)
      nr = convert_result_dataframe_string_to_list(nr)
      if i > 3:
        i = i - 4  
      linestyle = linestyles[i]

      
      lbl =labels[j]
      i +=1
      plt.plot(tpr,nr, label=lbl, linestyle=linestyle)
  total_noise_events = int(np.max(nr))  
  ax.legend(title=label_title)
  y_label = f'Noise reduction factor. Total {total_noise_events} noise events.'
  plt.ylabel(y_label)
  plt.xlabel('Efficiency/True Positive Rate')
  plt.title('Signal efficiency vs. noise reduction factor') 
  plt.semilogy(True)
  plt.xlim(x_low_lim,1)
  plt.grid()
  plt.tight_layout()  
  plt.savefig(save_path)  
  plt.show()
  plt.cla()

def directory(path , ending):
    list_dir = os.listdir(path)
    count = 0
    for file in list_dir:
        if file.endswith(ending):
            count += 1
    return count  

def count_models(dir_path='/home/halin/Autoencoder/Models/', ending = '.h5', start_folder=101, end_folder=192):
  count = 0
  for folder in range(101,192):
      path = dir_path + f'CNN_{folder}'
      count += directory(path, ending)
  print(count)
  return count

def loss_values_from_latent_space(signal_pred_values, noise_pred_values):

  """
    This function calculates the loss value based on the results from the 
    encoder part in the latent space. It calculates the midel point from the
    noise data and calculates the distance to that middle point for all values, 
    noise and signals. Bigger distance bigger loss value.
    Arguments: signal_pred_values, are the values from signals from the encoder
                                  in the latent space
               noise_pred_values, same as above
               
    Return: signal_loss, noise_loss                              

  """
  (data_points, dim) = noise_pred_values.shape
  x_noise = [0]*dim
  x_signal = [0]*dim
  for i in range(0,dim):
      x_noise[i] = noise_pred_values[:,i]
      x_signal[i] = signal_pred_values[:,i]
  x_middle_point_noise = [0]*dim
  for i in range(0,dim):
      x_middle_point_noise[i] = sum(x_noise[i])/len(x_noise[i])
  noise_loss = 0
  signal_loss = 0    
  for i in range(0,dim):
      noise_loss += (x_noise[i] - x_middle_point_noise[i])**2
      signal_loss += (x_signal[i] - x_middle_point_noise[i])**2
  noise_loss = noise_loss/len(noise_loss)
  signal_loss = signal_loss/len(signal_loss)
  return signal_loss, noise_loss

def plot_hist_example():
    signal_loss = np.load('/home/halin/Autoencoder/Models/test_models/signal_loss.npy')
    noise_loss = np.load('/home/halin/Autoencoder/Models/test_models/noise_loss.npy')
    save_path = '/home/halin/Autoencoder/Pictures/Example'
    hist(path=save_path, signal_loss=signal_loss,noise_loss=noise_loss)
