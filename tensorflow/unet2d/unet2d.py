import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, Activation
from tensorflow.keras.layers import Conv2D, MaxPool2D, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
import numpy as np

#%% Parameters
INPUT_SIZE = 64 # Input feature width/height
INPUT_CHANNEL = 1
OUTPUT_SIZE = 64 # Output feature width/height 
OUTPUT_CHANNEL = 1
OUTPUT_CLASSES = 1 # Number of output classes in dataset
base_filt = 32
dropout_rate = 0.15


