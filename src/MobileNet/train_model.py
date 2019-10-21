from __future__ import print_function
import sys

sys.path.append("..")
import tensorflow as tf
import numpy as np
import shutil
import math
import time
from PIL import Image
import random
import os
import os.path as path
p = path.abspath(path.join(__file__, "../../.."))
sys.path.append(p)
import src.train_utils as tu
from rs_net_ch import rs_net_ch

from tensorflow.python.keras.optimizers import Adam, RMSprop
from tensorflow.python.keras.backend import int_shape
from tensorflow.python.keras.models import Model, save_model, load_model
from tensorflow.python.keras.optimizers import Adam, RMSprop
from tensorflow.python.keras.metrics import top_k_categorical_accuracy,categorical_accuracy
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler
from tensorflow.python.keras.utils import multi_gpu_model
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import clip_ops

# Mahya
'''IMAGENET_PATH = '/HD1/'
TRAIN_PATH = 'Train/'
VAL_2_PATH = 'Val_2/'
META_FILE = 'dev_kit/ILSVRC2012_devkit_t12/data/meta.mat'
CONFIG_PATH = os.getcwd()
VALID_TIME_MINUTE = 5'''

# Pitagyro
'''IMAGENET_PATH = '/HD1/'
TRAIN_PATH = 'ILSVRC2012_img_train/'
VAL_2_PATH = 'Val_2/'
META_FILE = 'ILSVRC2012_devkit_t12/data/meta.mat'
CONFIG_PATH = os.getcwd()
VALID_TIME_MINUTE = 5'''

# MC
IMAGENET_PATH = '/HD1/'
TRAIN_PATH = 'train/'
VAL_2_PATH = 'val/'
META_FILE = 'ILSVRC2012_devkit_t12/data/meta.mat'
CONFIG_PATH = os.getcwd()
VALID_TIME_MINUTE = 5

def main():
    model_path = './L10_25p_v1/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    # 5 classes
    # ofms = [32, 43, 107, 116, 197, 197, 197, 256, 512, 512, 5]
    # 10 classes
    ofms = [32, 64, 116, 116, 197, 197, 213, 301, 569, 569, 10]
    # 15 classes
    # ofms = [32, 43, 107, 116, 213, 213, 213, 256, 512, 512, 15]
    print(len(ofms))
    with open(model_path + '/ofms.txt', 'w') as f:
        for ofm in ofms:
            f.write("%s\n" % ofm)

    num_classes = ofms[-1]  # number of classes
    first_class = 0
    last_class = 49
    num_epochs = 1000
    b_size = 64  # Batch size
    val_b_size = 64
    validation_period = 10
    augment_data = True  # Augment data or not
    load_weights = False
    use_tfrecord_format = False
    use_aux = False

    g_csn = rs_net_ch(num_classes=num_classes, ofms=ofms)  # Create model

    adam = Adam()
    # g_csn.compile(optimizer=adam, loss=[focal_loss(alpha=.25, gamma=2),
    # focal_loss(alpha=.25, gamma=2)], metrics=[categorical_accuracy],loss_weights=[1.0,0.3])

    g_csn.compile(optimizer=adam, loss=[tu.focal_loss(alpha=.25, gamma=2)],
                  metrics=[categorical_accuracy, tu.global_accuracy, tu.local_accuracy])
    if load_weights is True:
        g_csn = tu.load_model_npy(g_csn, 'weights.npy')

    # Train model
    if use_tfrecord_format:
        g_csn_trained = tu.fit_model_tfr(g_csn, num_classes, batch_size=b_size, \
                                         val_batch_size=val_b_size, val_period=validation_period, op_type='adam',
                                         model_path=model_path, \
                                         dataset_path=IMAGENET_PATH, train_path=TRAIN_PATH, \
                                         val_path=VAL_2_PATH, format='tfrecord', tfr_path='./', meta_path=META_FILE, \
                                         tb_logpath=model_path + "/logs",
                                         num_epochs=num_epochs, \
                                         augment=augment_data, multi_outputs=use_aux)
    else:
        g_csn_trained = tu.fit_model(g_csn, num_classes, first_class, last_class, batch_size=b_size, \
                                     val_batch_size=val_b_size, val_period=validation_period, op_type='adam',
                                     model_path=model_path, \
                                     imagenet_path=IMAGENET_PATH, train_path=TRAIN_PATH, \
                                     val_path=VAL_2_PATH, meta_path=META_FILE, \
                                     tb_logpath=model_path + "/logs", config_path=CONFIG_PATH,
                                     num_epochs=num_epochs, \
                                     augment=augment_data, multi_outputs=use_aux)

    shutil.move("selected_dirs.txt", model_path)
    '''shutil.move("weights.npy", model_path)
    shutil.move("weights.hdf5", model_path)
    shutil.move("rs_model_final.h5", model_path)'''


if __name__ == '__main__':
    main()
