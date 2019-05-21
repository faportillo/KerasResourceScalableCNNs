from __future__ import print_function

import sys

sys.path.append("..")
import tensorflow as tf
import numpy as np

import src.train_utils as tu

import os
import os.path as path
p = path.abspath(path.join(__file__, "../../../.."))
sys.path.append(p)
import math
import time
from PIL import Image
import random
from tensorflow.python.keras.layers import Input, Dense, Convolution2D, \
    MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, Reshape, \
    Activation, Concatenate, Layer, Lambda, BatchNormalization
from tensorflow.python.keras.backend import int_shape
from tensorflow.python.keras.models import Model, save_model, load_model
from tensorflow.python.keras.optimizers import Adam, RMSprop
from tensorflow.python.keras.metrics import top_k_categorical_accuracy,categorical_accuracy
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler
from tensorflow.python.keras.utils import multi_gpu_model
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import clip_ops

'''
    DistillNet - GoogLeNet version
    Check the 'Train and Validate Model' Section to modify the model
'''

'''
    Local Response Norm 
'''
def LRN(x):
    return tf.nn.local_response_normalization(x)

'''
    Define GoogLeNet Model
'''
def rs_net_ch(num_classes, ofms):
    assert len(ofms) == 58, "Number of ofms doesn't match model structure for GoogLeNet"
    '''
        X : input
        num_classes : number of desired classes for training    '''
    model_name = 'rs_net_ch'
    model_path = 'GoogeLeNet_Distill'

    input = Input(shape=(227, 227, 3))

    conv1_7x7_s2 = Convolution2D(ofms[0], 7, strides=(2, 2), padding='same', \
                                 activation=None, name='conv1/7x7_s2')(input)
    conv1_7x7_s2_bn = BatchNormalization(scale=False,name='conv1_7x7_s2_bn')(conv1_7x7_s2)
    conv1_7x7_s2_relu = Activation('relu',name='conv1_7x7_s2_relu')(conv1_7x7_s2_bn)
    print(str(int_shape(conv1_7x7_s2_relu)))

    pool1_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), \
                                padding='same', name='pool1/3x3_s2')(conv1_7x7_s2_relu)

    #pool1_norm1 = Lambda(LRN, name='pool1/norm1')(pool1_3x3_s2)
    print(str(int_shape(pool1_3x3_s2)))

    conv2_3x3_reduce = Convolution2D(ofms[1], 1, 1, padding='same', \
                                     activation='relu', name='conv2/3x3_reduce')(pool1_3x3_s2)

    conv2_3x3_reduce_bn = BatchNormalization(scale=False,name='conv2_3x3_reduce_bn')(conv2_3x3_reduce)
    conv2_3x3_reduce_relu = Activation('relu', name='conv2_7x7_s2_relu')(conv2_3x3_reduce_bn)

    conv2_3x3 = Convolution2D(ofms[2], 3, padding='same', activation='relu', \
                              name='conv2/3x3')(conv2_3x3_reduce_relu)
    conv2_3x3_bn = BatchNormalization(scale=False, name='conv2_3x3_bn')(conv2_3x3)
    conv2_3x3_relu = Activation('relu', name='conv2_3x3_relu')(conv2_3x3_bn)
    print(str(int_shape(conv2_3x3_relu)))

    #conv2_norm2 = Lambda(LRN, name='conv2/norm2')(conv2_3x3)
    pool2_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), \
                                padding='valid', name='pool2/3x3_s2')(conv2_3x3_relu)
    print(str(int_shape(pool2_3x3_s2)))

    '''
        Inception 3a
    '''
    inception_3a_1x1 = Convolution2D(ofms[3], 1, 1, padding='same', \
                                     activation='relu', name='inception_3a/1x1')(pool2_3x3_s2)
    inception_3a_1x1_bn = BatchNormalization(scale=False, name='inception_3a_1x1_bn')(inception_3a_1x1)
    inception_3a_1x1_relu = Activation('relu', name='inception_3a_1x1_relu')(inception_3a_1x1_bn)



    inception_3a_3x3_reduce = Convolution2D(ofms[4], 1, 1, padding='same', \
                                            activation='relu', name='inception_3a/3x3_reduce')(pool2_3x3_s2)
    inception_3a_5x5_reduce = Convolution2D(ofms[5], 1, 1, padding='same', \
                                            activation='relu', name='inception_3a/5x5_reduce')(pool2_3x3_s2)
    inception_3a_3x3_reduce_bn = BatchNormalization(scale=False, name='inception_3a_3x3_reduce_bn')(inception_3a_3x3_reduce)
    inception_3a_3x3_reduce_relu = Activation('relu', name='inception_3a_3x3_reduce_relu')(inception_3a_3x3_reduce_bn)


    inception_3a_3x3 = Convolution2D(ofms[6], 3, padding='same', activation='relu', \
                                     name='inception_3a/3x3')(inception_3a_3x3_reduce_relu)
    inception_3a_3x3_bn = BatchNormalization(scale=False, name='inception_3a_3x3_bn')(
        inception_3a_3x3)
    inception_3a_3x3_relu = Activation('relu', name='inception_3a_3x3_relu')(inception_3a_3x3_bn)



    inception_3a_5x5_reduce_bn = BatchNormalization(scale=False, name='inception_3a_5x5_reduce_bn')(
        inception_3a_5x5_reduce)
    inception_3a_5x5_reduce_relu = Activation('relu', name='inception_3a_5x5_reduce_relu')(inception_3a_5x5_reduce_bn)


    inception_3a_5x5 = Convolution2D(ofms[7], 5, padding='same', activation='relu', \
                                     name='inception_3a/5x5')(inception_3a_5x5_reduce_relu)
    inception_3a_5x5_bn = BatchNormalization(scale=False, name='inception_3a_5x5_bn')(
        inception_3a_5x5)
    inception_3a_5x5_relu = Activation('relu', name='inception_3a_5x5_relu')(inception_3a_5x5_bn)


    inception_3a_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), \
                                     padding='same', name='inception_3a/pool')(pool2_3x3_s2)

    inception_3a_pool_proj = Convolution2D(ofms[8], 1, 1, padding='same', \
                                           activation='relu', name='inception_3a/pool_proj')(inception_3a_pool)
    inception_3a_pool_proj_bn = BatchNormalization(scale=False, name='inception_3a_pool_proj_bn')(
        inception_3a_pool_proj)
    inception_3a_pool_proj_relu = Activation('relu', name='inception_3a_pool_proj_relu')(inception_3a_pool_proj_bn)

    inception_3a_output = Concatenate(axis=3, name='inception_3a/output') \
        ([inception_3a_1x1_relu, inception_3a_3x3_relu, inception_3a_5x5_relu, \
          inception_3a_pool_proj_relu])
    print("INCEPTION3a OUT:"+str(int_shape(inception_3a_output)))

    '''
        Inception 3b
    '''

    inception_3b_1x1 = Convolution2D_BN(inception_3a_output,ofms[9], 1, 1, padding='same', \
                                      name='inception_3b/1x1')

    inception_3b_3x3_reduce = Convolution2D_BN(inception_3a_output,ofms[10], 1, 1, padding='same', \
                                             name='inception_3b/3x3_reduce')
    inception_3b_5x5_reduce = Convolution2D_BN(inception_3a_output,ofms[11], 1, 1, padding='same', \
                                             name='inception_3b/5x5_reduce')

    inception_3b_3x3 = Convolution2D_BN(inception_3b_3x3_reduce,ofms[12], 3, padding='same', \
                                      name='inception_3b/3x3')


    inception_3b_5x5 = Convolution2D_BN(inception_3b_5x5_reduce,ofms[13], 5, padding='same', \
                                     name='inception_3b/5x5')

    inception_3b_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), \
                                     padding='same', name='inception_3b/pool')(inception_3a_output)

    inception_3b_pool_proj = Convolution2D_BN(inception_3b_pool,ofms[14], 1, 1, padding='same', \
                                            name='inception_3b/pool_proj')

    inception_3b_output = Concatenate(axis=3, name='inception_3b/output') \
        ([inception_3b_1x1, inception_3b_3x3, inception_3b_5x5, \
          inception_3b_pool_proj])
    print("INCEPTIOn3b OUT"+str(int_shape(inception_3b_output)))

    pool3_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), \
                                padding='same', name='pool3/3x3_s2')(inception_3b_output)
    print(str(int_shape(pool3_3x3_s2)))

    '''
        Inception 4a
    '''
    inception_4a_1x1 = Convolution2D_BN(pool3_3x3_s2,ofms[15], 1, 1, padding='same', \
                                     activation='relu', name='inception_4a/1x1')

    inception_4a_3x3_reduce = Convolution2D_BN(pool3_3x3_s2,ofms[16], 1, 1, padding='same', \
                                            activation='relu', name='inception_4a/3x3_reduce')
    inception_4a_5x5_reduce = Convolution2D_BN(pool3_3x3_s2,ofms[17], 1, 1, padding='same', \
                                            activation='relu', name='inception_4a/5x5_reduce')

    inception_4a_3x3 = Convolution2D_BN(inception_4a_3x3_reduce,ofms[18], 3, padding='same', \
                                     activation='relu', name='inception_4a/3x3')

    inception_4a_5x5 = Convolution2D_BN(inception_4a_5x5_reduce,ofms[19], 5, padding='same', activation='relu', \
                                     name='inception_4a/5x5')

    inception_4a_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), \
                                     padding='same', name='inception_4a/pool')(pool3_3x3_s2)

    inception_4a_pool_proj = Convolution2D_BN(inception_4a_pool,ofms[20], 1, 1, padding='same', \
                                           activation='relu', name='inception_4a/pool_proj')

    inception_4a_output = Concatenate(axis=3, name='inception_4a/output') \
        ([inception_4a_1x1, inception_4a_3x3, inception_4a_5x5, \
          inception_4a_pool_proj])
    print("INCEPTION4a OUT"+str(int_shape(inception_4a_output)))
    '''
        Inception 4b
    '''
    inception_4b_1x1 = Convolution2D_BN(inception_4a_output,ofms[21], 1, 1, padding='same', \
                                     activation='relu', name='inception_4b/1x1')

    inception_4b_3x3_reduce = Convolution2D_BN(inception_4a_output,ofms[22], 1, 1, padding='same', \
                                            activation='relu', name='inception_4b/3x3_reduce')
    inception_4b_5x5_reduce = Convolution2D_BN(inception_4a_output,ofms[23], 1, 1, padding='same', \
                                            activation='relu', name='inception_4b/5x5_reduce')

    inception_4b_3x3 = Convolution2D_BN(inception_4b_3x3_reduce,ofms[24], 3, padding='same', \
                                     activation='relu', name='inception_4b/3x3')

    inception_4b_5x5 = Convolution2D_BN(inception_4b_5x5_reduce,ofms[25], 5, padding='same', activation='relu', \
                                     name='inception_4b/5x5')

    inception_4b_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), \
                                     padding='same', name='inception_4b/pool')(inception_4a_output)

    inception_4b_pool_proj = Convolution2D_BN(inception_4b_pool,ofms[26], 1, 1, padding='same', \
                                           activation='relu', name='inception_4b/pool_proj')

    inception_4b_output = Concatenate(axis=3, name='inception_4b/output') \
        ([inception_4b_1x1, inception_4b_3x3, inception_4b_5x5, \
          inception_4b_pool_proj])
    print("INCEPTION4b OUT"+str(int_shape(inception_4b_output)))

    branch_num = 0



    inception_4c_1x1 = Convolution2D_BN(inception_4b_output, ofms[27], 1, 1, padding='same', \
                                        activation='relu', name=str(branch_num) +'inception_4c_1x1')

    inception_4c_3x3_reduce = Convolution2D_BN(inception_4b_output, ofms[28], 1, 1, padding='same', \
                                               activation='relu', name=str(branch_num) +'inception_4c_3x3_reduce')
    inception_4c_5x5_reduce = Convolution2D_BN(inception_4b_output, ofms[29], 1, 1, padding='same', \
                                               activation='relu', name=str(branch_num) +'inception_4c/5x5_reduce')
    inception_4c_3x3 = Convolution2D_BN(inception_4c_3x3_reduce, ofms[30], 3, padding='same', activation='relu', \
                                     name=str(branch_num) + 'inception_4c/3x3')

    inception_4c_5x5 = Convolution2D_BN(inception_4c_5x5_reduce, ofms[31], 5, padding='same', activation='relu', \
                                     name=str(branch_num) + 'inception_4c/5x5')

    inception_4c_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), \
                                     padding='same', name=str(branch_num) + 'inception_4c/pool') \
        (inception_4b_output)

    inception_4c_pool_proj = Convolution2D_BN(inception_4c_pool, ofms[32], 1, 1, padding='same', \
                                           activation='relu', name=str(branch_num) + 'inception_4c/pool_proj') \


    inception_4c_output = Concatenate(axis=3, name=str(branch_num) + \
                                                   'inception_4c/output')([inception_4c_1x1,\
                                                                           inception_4c_3x3, inception_4c_5x5, \
                                                                           inception_4c_pool_proj])
    print("INCEPTION4c OUT"+str(int_shape(inception_4c_output)))
    '''
        Inception 4d
    '''
    inception_4d_1x1 = Convolution2D_BN(inception_4c_output, ofms[33], 1, 1, padding='same', \
                                        activation='relu', name=str(branch_num) +'inception_4d_1x1')

    inception_4d_3x3_reduce = Convolution2D_BN(inception_4c_output, ofms[34], 1, 1, padding='same', \
                                               activation='relu', name=str(branch_num) +'inception_4d_3x3_reduce')
    inception_4d_5x5_reduce = Convolution2D_BN(inception_4c_output, ofms[35], 1, 1, padding='same', \
                                               activation='relu', name=str(branch_num) +'inception_4d_5x5_reduce')
    inception_4d_3x3 = Convolution2D_BN(inception_4d_3x3_reduce, ofms[36], 3, padding='same', activation='relu', \
                                        name=str(branch_num) + 'inception_4d_3x3')

    inception_4d_5x5 = Convolution2D_BN(inception_4d_5x5_reduce, ofms[37], 5, padding='same', activation='relu', \
                                        name=str(branch_num) + 'inception_4d_5x5')

    inception_4d_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), \
                                     padding='same', name=str(branch_num) + 'inception_4d_pool') \
        (inception_4c_output)

    inception_4d_pool_proj = Convolution2D_BN(inception_4d_pool, ofms[38], 1, 1, padding='same', \
                                              activation='relu', name=str(branch_num) + 'inception_4d_pool_proj')
    inception_4d_output = Concatenate(axis=3, name=str(branch_num) + 'inception_4d_output')([inception_4d_1x1, \
                                                                                   inception_4d_3x3, inception_4d_5x5, \
                                                                                   inception_4d_pool_proj])

    print("INCEPTION4d OUT"+str(int_shape(inception_4d_output)))
    '''
        Inception 4e
    '''
    inception_4e_1x1 = Convolution2D_BN(inception_4d_output, ofms[39], 1, 1, padding='same', \
                                        activation='relu', name=str(branch_num) +'inception_4e_1x1')

    inception_4e_3x3_reduce = Convolution2D_BN(inception_4d_output, ofms[40], 1, 1, padding='same', \
                                               activation='relu', name=str(branch_num) +'inception_4e_3x3_reduce')
    inception_4e_5x5_reduce = Convolution2D_BN(inception_4d_output, ofms[41], 1, 1, padding='same', \
                                               activation='relu', name=str(branch_num) +'inception_4e_5x5_reduce')
    inception_4e_3x3 = Convolution2D_BN(inception_4e_3x3_reduce, ofms[42], 3, padding='same', activation='relu', \
                                        name=str(branch_num) + 'inception_4e_3x3')

    inception_4e_5x5 = Convolution2D_BN(inception_4e_5x5_reduce, ofms[43], 5, padding='same', activation='relu', \
                                        name=str(branch_num) + 'inception_4e_5x5')

    inception_4e_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), \
                                     padding='same', name=str(branch_num) + 'inception_4e_pool') \
        (inception_4d_output)

    inception_4e_pool_proj = Convolution2D_BN(inception_4e_pool, ofms[44], 1, 1, padding='same', \
                                              activation='relu', name=str(branch_num) + 'inception_4e_pool_proj')
    inception_4e_output = Concatenate(axis=3, name=str(branch_num) + 'inception_4e_output')([inception_4e_1x1, \
                                                                                             inception_4e_3x3,
                                                                                             inception_4e_5x5, \
                                                                                             inception_4e_pool_proj])
    print("INCEPTION4e OUT"+str(int_shape(inception_4e_output)))

    pool4_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), \
                                padding='same', name=str(branch_num) + 'pool4/3x3_s2') \
        (inception_4e_output)
    print(str(int_shape(pool4_3x3_s2)))
    '''
        Inception 5a
    '''
    inception_5a_1x1 = Convolution2D_BN(pool4_3x3_s2, ofms[45], 1, 1, padding='same', \
                                                activation='relu', name=str(branch_num) +'inception_5a_1x1')

    inception_5a_3x3_reduce = Convolution2D_BN(pool4_3x3_s2, ofms[46], 1, 1, padding='same', \
                                               activation='relu', name=str(branch_num) +'inception_5a_3x3_reduce')
    inception_5a_5x5_reduce = Convolution2D_BN(pool4_3x3_s2, ofms[47], 1, 1, padding='same', \
                                               activation='relu', name=str(branch_num) +'inception_5a_5x5_reduce')

    inception_5a_3x3 = Convolution2D_BN(inception_5a_3x3_reduce, ofms[48], 3, padding='same', activation='relu', \
                                     name=str(branch_num) + 'inception_5a_3x3')

    inception_5a_5x5 = Convolution2D_BN(inception_5a_5x5_reduce, ofms[49], 5, padding='same', activation='relu', \
                                     name=str(branch_num) + 'inception_5a_5x5')

    inception_5a_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), \
                                     padding='same', name=str(branch_num) + 'inception_5a_pool') \
        (pool4_3x3_s2)

    inception_5a_pool_proj = Convolution2D_BN(inception_5a_pool, ofms[50], 1, 1, padding='same', \
                                           activation='relu', name=str(branch_num) + 'inception_5a_pool_proj') \


    inception_5a_output = Concatenate(axis=3, name=str(branch_num) + \
                                                   'inception_5a_output')([inception_5a_1x1,inception_5a_3x3,\
                                                                           inception_5a_5x5,inception_5a_pool_proj])
    print("INCEPTION5a OUT"+str(int_shape(inception_5a_output)))
    '''
        Inception 5b
    '''
    inception_5b_1x1 = Convolution2D_BN(inception_5a_output, ofms[51], 1, 1, padding='same', \
                                        activation='relu', name=str(branch_num) +'inception_5b_1x1')

    inception_5b_3x3_reduce = Convolution2D_BN(inception_5a_output, ofms[52], 1, 1, padding='same', \
                                               activation='relu', name=str(branch_num) +'inception_5b_3x3_reduce')
    inception_5b_5x5_reduce = Convolution2D_BN(inception_5a_output, ofms[53], 1, 1, padding='same', \
                                               activation='relu', name=str(branch_num) +'inception_5b_5x5_reduce')

    inception_5b_3x3 = Convolution2D_BN(inception_5b_3x3_reduce, ofms[54], 3, padding='same', activation='relu', \
                                        name=str(branch_num) + 'inception_5b_3x3')

    inception_5b_5x5 = Convolution2D_BN(inception_5b_5x5_reduce, ofms[55], 5, padding='same', activation='relu', \
                                        name=str(branch_num) + 'inception_5b_5x5')

    inception_5b_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), \
                                     padding='same', name=str(branch_num) + 'inception_5b_pool') \
        (inception_5a_output)

    inception_5b_pool_proj = Convolution2D_BN(inception_5b_pool, ofms[56], 1, 1, padding='same', \
                                              activation='relu', name=str(branch_num) + 'inception_5b_pool_proj')

    inception_5b_output = Concatenate(axis=3, name=str(branch_num) + 'inception_5b_output')\
        ([inception_5b_1x1, inception_5b_3x3, inception_5b_5x5,inception_5b_pool_proj])

    print("INCEPTION5b OUT"+str(int_shape(inception_5b_output)))

    pool5_7x7_s1 = AveragePooling2D(pool_size=(7, 7), strides=(1, 1), \
                                    name=str(branch_num) + 'pool5/7x7_s2')(inception_5b_output)
    print(str(int_shape(pool5_7x7_s1)))

    loss3_flat = Flatten(name=str(branch_num) + '_flatten')(pool5_7x7_s1)
    print(str(int_shape(loss3_flat)))
    # pool5_drop_7x7_s1 = Dropout(dropout, \
    # name=str(branch_num)+'_dropout')(loss3_flat)
    # print(str(int_shape(pool5_drop_7x7_s1)))
    result = Dense(num_classes, name=str(branch_num) + '_loss3/classifier')(loss3_flat)


    '''
        Auxiliary Loss
    '''
    aux_loss_avg_pool = AveragePooling2D(pool_size=(5, 5), strides=(3, 3),\
                                        name='auxloss/ave_pool')\
                                        (inception_4b_output)
    aux_loss_conv = Convolution2D_BN(aux_loss_avg_pool,128, 1, 1,\
                                     padding='same', \
                                     activation='relu',\
                                    name='auxloss/conv')
    print("AUX LOSS CONV SHAPE"+str(int_shape(aux_loss_conv)))
    aux_loss_flat = Flatten()(aux_loss_conv)
    print("AUX FLAT SHAPE"+str(int_shape(aux_loss_flat)))
    aux_loss_fc = Dense(1024, activation='relu', name=\
                                                    'auxloss/fc')(aux_loss_flat)
    aux_drop_fc = Dropout(0.4)(aux_loss_fc)
    aux_classifier = Dense(num_classes, name= \
                                      'auxloss/classifier')(aux_drop_fc)
    aux_classifier_act = Activation('softmax', name='prob_aux')(aux_classifier)
    classes = []
    # for c in range(num_branches):
    #     cls = inception_class_branch(inception_4b_output, branch_num=c)
    #     classes.append(cls)

    # main_classifier = Concatenate(axis=-1)(classes)
    print("AUX OUT SHAPE"+str(int_shape(aux_classifier)))
    print("MAIN OUT SHAPE"+str(int_shape(result)))
    main_classifier_act = Activation('softmax', name='prob_main') \
        (result)

    print("OUTPUT: " + str(int_shape(main_classifier_act)))
    googlenet = Model(inputs=input, outputs=[main_classifier_act,aux_classifier_act])

    return googlenet

def Convolution2D_BN(x,filters,kernel_size,stride=(1,1),padding='same',\
                     activation='relu',name=None):
    x = Convolution2D(filters,kernel_size,stride,padding=padding,name=name)(x)
    x = BatchNormalization(scale=False,name=name+'_bn')(x)
    x = Activation(activation,name=name+'_relu')(x)
    return x
