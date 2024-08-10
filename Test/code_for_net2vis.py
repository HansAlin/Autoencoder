# You can freely modify this file.
# However, you need to have a function that is named get_model and returns a Keras Model.
import keras as k
from keras import models
from keras import layers
from keras import utils

def get_model():
    img_height = 1
    img_width = 100
    img_channels = 1

    input_shape = (img_width, img_channels)
    img_input = k.Input(shape=input_shape)
    layer = layers.Conv1D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(img_input)
    
    layer = layers.MaxPooling1D(pool_size=2)(layer)
    
    layer = layers.Conv1D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layer)
    
    layer = layers.MaxPooling1D(pool_size=2)(layer)
    
    layer = layers.Dense(8)(layer)
    
    layer = layers.UpSampling1D()(layer)
    
    layer = layers.Conv1D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layer)
    
    layer = layers.UpSampling1D()(layer)
    
    layer = layers.Conv1D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layer)
    
    layer = layers.Dense(1)(layer)
    

    model = models.Model(img_input, layer)

    return model