from __future__ import print_function
import sys

sys.path.append("..")
import tensorflow as tf
import numpy as np
import train_utils as tu

import os
import math
import time
import re
from PIL import Image
from operator import itemgetter
from collections import OrderedDict
import random
from tensorflow.python.keras.layers import Input, Dense, Convolution2D, \
    MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, Reshape, \
    Activation, Concatenate, Layer, Lambda, BatchNormalization
from tensorflow.python.keras.backend import int_shape
from tensorflow.python.keras.models import Model, save_model, load_model
from tensorflow.python.keras.optimizers import Adam, RMSprop
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.metrics import top_k_categorical_accuracy,categorical_accuracy
from google_csn_keras import google_csn, focal_loss

def remove_channels(model, layer, channel_list):
    '''
    USED FOR PRUNING weights on a specific layer
    :param model: Keras model - Model object
    :param layer: Desired layer number - int
    :param channel_list: List of channels to be removed for pruning
    :return: Model with reduced number of channels in desired layer
    '''
    # Get weights and biases for desired layer
    layer_weights = model.layers[layer].get_weights()[0]
    print(layer_weights.shape)
    layer_biases = model.layers[layer].get_weights()[1]
    reduced_weights = layer_weights
    reduced_biases = layer_biases
    reduced_weights[:, :, :, channel_list] = 0
    reduced_biases[channel_list] = 0
    pruned_weights = [reduced_weights, reduced_biases]
    model.layers[layer].set_weights(pruned_weights)

    return model

def prune_by_std(model, s=0.25):
    """
    According to https://arxiv.org/pdf/1506.02626.pdf
    (Learning both Weights and Connections for Efficient Neural Networks),
    'The pruning threshold is chosen as a quality parameter multiplied
    by the standard deviation of a layer’s weights'
    :param s: Sensitivity factor
    :return: model with pruned weights
    """

    layer_num = 0
    for layer in model.layers:
        if ('Conv2D' in layer.__class__.__name__ or 'Dense' in layer.__class__.__name__)\
                and 'aux' not in layer.name:
            print(layer.__class__.__name__)
            layer_weights = model.layers[layer_num].get_weights()[0]
            layer_biases = model.layers[layer_num].get_weights()[1]
            threshold = np.std(layer_weights) * s
            weights_mask = np.where(abs(layer_weights) < threshold, 0, 1)
            bias_mask = np.where(abs(layer_biases) < threshold, 0, 1)
            pruned_weights = [np.multiply(weights_mask, layer_weights),
                              np.multiply(bias_mask, layer_biases)]
            model.layers[layer_num].set_weights(pruned_weights)

        layer_num+=1

    return model