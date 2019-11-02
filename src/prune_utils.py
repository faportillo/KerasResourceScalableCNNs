from __future__ import print_function
import sys

sys.path.append("..")
import tensorflow as tf
import numpy as np
import train_utils as tu
import eval_utils as eu

import os
import math
import time
import re
from PIL import Image
from operator import itemgetter
from collections import OrderedDict
from shutil import copyfile
from sys import exit
from os import path
import random
from tensorflow.python.keras.layers import Input, Dense, Convolution2D, \
    MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, Reshape, \
    Activation, Concatenate, Layer, Lambda, BatchNormalization
from tensorflow.python.keras.backend import int_shape
from tensorflow.python.keras.models import Model, save_model, load_model
from tensorflow.python.keras.optimizers import Adam, RMSprop
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.metrics import top_k_categorical_accuracy,categorical_accuracy
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, \
    Callback, LearningRateScheduler, TerminateOnNaN

import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.sparsity import keras as sparsity

'''
    ToDo: 
    - Need to have methods to evaluate local and global accuracy during
      inference. 
    
    - Need to have method that accepts model and prunes it based on desired 
      percent error or parameter count *Check Song Han paper to see how they
      determined stop to pruning.
'''

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

    total_pruned_weights = 0
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
            total_pruned_weights += np.count_nonzero(weights_mask == 0)
            total_pruned_weights += np.count_nonzero(bias_mask == 0)

        layer_num+=1

    return model, total_pruned_weights

def prune_model(model, model_type='googlenet_rs',
                num_classes=1000,
                batch_size=64,
                val_batch_size=50,
                loss_type='focal',
                op_type='adam',
                imagenet_path=None,
                model_path='./',
                train_path=None,
                val_path=None,
                tb_logpath='./logs',
                meta_path=None,
                num_epochs=1000,
                augment=True,
                multi_outputs=False,
                garbage_multiplier=1):

    '''
        Prune a resource-scalable model

    :param model:
    :param model_type: sees if its resource scalable model or a given Keras default model
    :param num_classes:
    :param batch_size:
    :param val_batch_size:
    :param loss_type:
    :param op_type:
    :param imagenet_path:
    :param model_path:
    :param train_path:
    :param val_path:
    :param tb_logpath:
    :param meta_path:
    :param num_epochs:
    :return:
    '''
    orig_train_img_path = os.path.join(imagenet_path, train_path)
    orig_val_img_path = os.path.join(imagenet_path, val_path)
    train_img_path = orig_train_img_path
    val_img_path = orig_val_img_path
    wnid_labels, _ = tu.load_imagenet_meta(os.path.join(imagenet_path, \
                                                     meta_path))

    # copy selected_dirs.txt to current directory if not already in it
    if not path.exists('selected_dirs.txt'):
        try:
            copyfile(model_path + 'selected_dirs.txt', './')
            print("File copied from %s" % model_path)
        except IOError as e:
            print("Unable to copy file. %s" % e)
            exit(1)
        except:
            print("Unexpected error:", sys.exc_info())
            exit(1)

    if model_type == 'googlenet_rs' or model_type == 'mobilenet_rs':
        new_training_path = os.path.join(imagenet_path, "GARBAGE_TRAIN_")
        new_validation_path = os.path.join(imagenet_path, "GARBAGE_VAL_")
        selected_classes = tu.create_garbage_links(num_classes, wnid_labels, train_img_path, \
                                                new_training_path, val_img_path, new_validation_path)
        tu.create_garbage_class_folder(selected_classes, wnid_labels,
                                    train_img_path, new_training_path,
                                    val_img_path, new_validation_path)
        train_img_path = new_training_path
        val_img_path = new_validation_path

    tb_callback = TensorBoard(log_dir=tb_logpath)
    termNaN_callback = TerminateOnNaN()

    callbacks = [tb_callback, termNaN_callback,
                 sparsity.UpdatePruningStep(),
                 sparsity.PruningSummaries(log_dir=tb_logpath, profile_batch=0)]

    if op_type == 'rmsprop':
        '''
            If the optimizer type is RMSprop, decay learning rate
            and append to callback list
        '''
        lr_decay_callback = ExpDecayScheduler(decay_params[0], \
                                              decay_params[1], decay_params[2])
        callback_list.append(lr_decay_callback)
    elif op_type == 'adam':
        print ('Optimizer: Adam')
    elif op_type == 'sgd':
        print('Optimizer: SGD')
        one_cycle = OneCycle(clrcm_params[0], clrcm_params[1], clrcm_params[2], \
                             clrcm_params[3], clrcm_params[4], clrcm_params[5])
        callback_list.append(one_cycle)
    else:
        # print ('Invalid Optimizer. Exiting...')
        exit()

    end_step = np.ceil(1.0 * ((num_classes-1) * 1300)+(1300*garbage_multiplier) / batch_size).astype(np.int32) * num_epochs
    orig_stdout = sys.stdout
    f = open('orig_model_summary.txt', 'w')
    sys.stdout = f
    print(model.summary())
    sys.stdout = orig_stdout
    f.close()
    print("Defining pruning schedule...")
    new_pruning_params = {'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.0,
                                                                       final_sparsity=0.50,
                                                                       begin_step=0,
                                                                       end_step=end_step,
                                                                       frequency=100)}

    pruned_model = sparsity.prune_low_magnitude(model, **new_pruning_params)
    print("Compiling model")
    if model_type == 'googlenet_rs': #resource-scalable googlenet
        if multi_outputs:
            if loss_type == 'focal':
                loss1 = tu.focal_loss(alpha=.25, gamma=2)
                loss2 = tu.focal_loss(alpha=.25, gamma=2)
            else:
                loss1 = tu.dual_loss(alpha=.25, gamma=2)
                loss2 = tu.dual_loss(alpha=.25, gamma=2)

            train_data, val_data = tu.imagenet_generator_multi(train_img_path, \
                                                            val_img_path, batch_size=batch_size, \
                                                            do_augment=augment)

            pruned_model.compile(optimizer=op_type, loss=[loss1, loss2],
                          metrics=[categorical_accuracy, tu.global_accuracy, tu.local_accuracy],
                          loss_weights=[1.0, 0.3])
        else:
            if loss_type == 'focal':
                loss = tu.focal_loss(alpha=.25, gamma=2)
            else:
                loss = tu.dual_loss(alpha=.25, gamma=2)

            train_data, val_data = tu.imagenet_generator(train_img_path, \
                                                               val_img_path,
                                                               batch_size=batch_size, \
                                                               do_augment=augment)
            pruned_model.compile(optimizer=op_type, loss=[loss],
                                 metrics=[categorical_accuracy, tu.global_accuracy, tu.local_accuracy])

    elif model_type == 'mobilenet_rs':
        if loss_type == 'focal':
            loss = tu.focal_loss(alpha=.25, gamma=2)
        else:
            loss = tu.dual_loss(alpha=.25, gamma=2)

        train_data, val_data = tu.imagenet_generator(train_img_path, \
                                                     val_img_path,
                                                     batch_size=batch_size, \
                                                     do_augment=augment)
        pruned_model.compile(optimizer=op_type, loss=[loss],
                             metrics=[categorical_accuracy, tu.global_accuracy, tu.local_accuracy])

    elif model_type == 'googlenet': #Vanilla googlenet for 1000 classes
        pruned_model.compile(optimizer=op_type,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        train_data, val_data = tu.imagenet_generator(train_img_path, \
                                                        val_img_path, batch_size=batch_size, \
                                                        do_augment=augment)
    else:
        pruned_model.compile(optimizer=op_type,
                             loss='categorical_crossentropy',
                             metrics=['accuracy'])

        train_data, val_data = tu.imagenet_generator(train_img_path, \
                                                        val_img_path, batch_size=batch_size, \
                                                        do_augment=augment)
    print("Fitting model")
    pruned_model.fit_generator(train_data, epochs=num_epochs, \
                        steps_per_epoch=int(((num_classes-1) * 1300)+(1300*garbage_multiplier)) / batch_size, \
                        validation_data=val_data, \
                        validation_steps= \
                            int(50000 / val_batch_size), \
                        verbose=2, callbacks=callbacks, workers=1,
                        use_multiprocessing=False)

    return pruned_model


