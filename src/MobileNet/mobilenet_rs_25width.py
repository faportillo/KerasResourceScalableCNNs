from __future__ import print_function

import sys

sys.path.append("..")
import tensorflow as tf
import numpy as np

import os
import os.path as path
p = path.abspath(path.join(__file__, "../../.."))
sys.path.append(p)
p = path.abspath(path.join(__file__, "../.."))
sys.path.append(p)
import src.utils.train_utils as tu
import math
import time
from PIL import Image
import random
from tensorflow.python.keras.layers import Input, Dense, Convolution2D, \
    MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, Reshape, \
    Activation, Concatenate, Layer, Lambda, BatchNormalization,\
    DepthwiseConv2D, GlobalAveragePooling2D
from tensorflow.python.keras.backend import int_shape
from tensorflow.python.keras.models import Model, save_model, load_model
from tensorflow.python.keras.optimizers import Adam, RMSprop
from tensorflow.python.keras.metrics import top_k_categorical_accuracy,categorical_accuracy
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler
from tensorflow.python.keras.utils import multi_gpu_model
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import clip_ops

'''
    ResourceScalable Net - MobileNet version
'''

'''
    Define MobileNet Model
'''
def mobilenet_rs_25width(ofms, num_classes=1000, alpha=1, rho=1):
    assert len(ofms) == 11, "Number of ofms doesn't match model structure for Mobilenet"
    '''
        X : input
        num_classes : number of desired classes for training    '''
    img_input = Input(shape=(round(rho*227), round(rho*227), 3))

    # Macrolayer 1
    x = Convolution2D(int(ofms[0] * alpha), (3, 3), strides=(2, 2), padding='same', use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # Macrolayer 2
    x = DepthwiseConv2D((1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(int(ofms[1] * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # Macrolayer 3
    x = DepthwiseConv2D((1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(int(ofms[2] * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # Macrolayer 4
    x = DepthwiseConv2D((1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(int(ofms[3] * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # Macrolayer 5
    x = DepthwiseConv2D((1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(int(ofms[4] * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # Macrolayer 6
    x = DepthwiseConv2D((1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(int(ofms[5] * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # Macrolayer 7
    x = DepthwiseConv2D((1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(int(ofms[6] * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # # Macrolayer 8
    for _ in range(5):
        x = DepthwiseConv2D((1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Convolution2D(int(ofms[7] * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    # Macrolayer 9
    x = DepthwiseConv2D((1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(int(ofms[8] * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # Macrolayer 10
    x = DepthwiseConv2D((1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(int(ofms[9] * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = GlobalAveragePooling2D()(x)
    out = Dense(num_classes, activation='softmax')(x)

    inputs = img_input

    model = Model(inputs, out, name='mobilenet')

    return model
