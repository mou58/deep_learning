'''
Reference: https://github.com/96imranahmed/3D-Unet

The above mentioned repo implemented a 3D U-Net using tensorflow (no Keras). I have changed it for 2d U-Net. Besides, this implementation supports Keras API. 
'''

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

#%% Build the model
def conv_batch_relu(tensor, filters, name, kernel = 3, stride = 1):
    conv = Conv2D(filters, kernel_size = kernel, strides = stride, padding = 'same',
                  kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev=0.1), 
                  kernel_regularizer = tf.keras.regularizers.l2(0.1), 
                  name=name)(tensor)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    return conv

def upconvolve(tensor, filters, name, kernel = 2, stride = 2, activation = None):
    conv = Conv2DTranspose(filters, kernel_size = kernel, strides = stride, padding = 'same', use_bias=False, 
                                      kernel_initializer = tf.keras.initializers.TruncatedNormal,  
                                      kernel_regularizer = tf.keras.regularizers.l2(0.1), name=name)(tensor)
    return conv
  
 
def centre_crop_and_concat(prev_conv, up_conv):
    # Needed if the padding is 'valid'
    p_c_s = prev_conv.get_shape()
    u_c_s = up_conv.get_shape()
    offsets =  np.array([0, (p_c_s[1] - u_c_s[1]) // 2, (p_c_s[2] - u_c_s[2]) // 2, 0], dtype = np.int32)
    size = np.array([-1, u_c_s[1], u_c_s[2], p_c_s[-1]], np.int32)
    prev_conv_crop = tf.slice(prev_conv, offsets, size)
    up_concat = tf.concat((prev_conv_crop, up_conv), -1)
    return up_concat


model_input = Input(shape=(INPUT_SIZE, INPUT_SIZE, INPUT_CHANNEL), name='input_img')
# Level zero
conv_0_1 = conv_batch_relu(model_input, base_filt, name='conv_0_1')
conv_0_2 = conv_batch_relu(conv_0_1, base_filt*2, name='conv_0_2')
# Level one
max_1_1 = MaxPool2D([2,2], [2,2], name='max_1_1')(conv_0_2) 
conv_1_1 = conv_batch_relu(max_1_1, base_filt*2, name='conv_1_1')
conv_1_2 = conv_batch_relu(conv_1_1, base_filt*4, name='conv_1_2')
conv_1_2 = Dropout(rate = dropout_rate, name='conv_1_2_dropout')(conv_1_2)
# Level two
max_2_1 = MaxPool2D([2,2], [2,2], name='max_2_1')(conv_1_2) 
conv_2_1 = conv_batch_relu(max_2_1, base_filt*4, name='conv_2_1')
conv_2_2 = conv_batch_relu(conv_2_1, base_filt*8, name='conv_2_2')
conv_2_2 = Dropout(rate = dropout_rate, name='conv_2_2_dropout')(conv_2_2)
# Level three
max_3_1 = MaxPool2D([2,2], [2,2], name='max_3_1')(conv_2_2) 
conv_3_1 = conv_batch_relu(max_3_1, base_filt*8, name='conv_3_1')
conv_3_2 = conv_batch_relu(conv_3_1, base_filt*16, name='conv_3_2')
conv_3_2 = Dropout(rate = dropout_rate, name='conv_3_2_dropout')(conv_3_2)
# Level two
up_conv_3_2 = upconvolve(conv_3_2, base_filt*16, kernel = 2, stride = [2,2], name='up_conv_3_2')  
concat_2_1 = centre_crop_and_concat(conv_2_2, up_conv_3_2)
conv_2_3 = conv_batch_relu(concat_2_1, base_filt*8, name='conv_2_3')
conv_2_4 = conv_batch_relu(conv_2_3, base_filt*8, name='conv_2_4')
conv_2_4 = Dropout(rate = dropout_rate, name='conv_2_4_dropout')(conv_2_4)
# Level one
up_conv_2_1 = upconvolve(conv_2_4, base_filt*8, kernel = 2, stride = [2,2], name='up_conv_2_1')
concat_1_1 = centre_crop_and_concat(conv_1_2, up_conv_2_1)
conv_1_3 = conv_batch_relu(concat_1_1, base_filt*4, name='conv_1_3')
conv_1_4 = conv_batch_relu(conv_1_3, base_filt*4, name='conv_1_4')
conv_1_4 = Dropout(rate = dropout_rate, name='conv_1_4_dropout')(conv_1_4)
# Level zero
up_conv_1_0 = upconvolve(conv_1_4, base_filt*4, kernel = 2, stride = [2,2], name='conv_1_0') 
concat_0_1 = centre_crop_and_concat(conv_0_2, up_conv_1_0)
conv_0_3 = conv_batch_relu(concat_0_1, base_filt*2, name='conv_0_3')
conv_0_4 = conv_batch_relu(conv_0_3, base_filt*2, name='conv_0_4')
conv_0_4 = Dropout(rate = dropout_rate, name='conv_0_4_dropout')(conv_0_4)
conv_out = Conv2D(OUTPUT_CLASSES, [1,1], [1,1], padding = 'same', name='conv_out')(conv_0_4)

unet2d = Model(inputs=model_input, outputs=conv_out, name='unet3d')

print(unet2d.summary())

