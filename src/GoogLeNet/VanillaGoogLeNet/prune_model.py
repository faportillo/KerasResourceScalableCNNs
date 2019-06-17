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
p = path.abspath(path.join(__file__, "../../../.."))
sys.path.append(p)
p = path.abspath(path.join(__file__, "../../.."))
sys.path.append(p)
import src.train_utils as tu
import src.prune_utils as pu
import src.eval_utils as eu
import src.GoogLeNet.VanillaGoogLeNet.inception_v1 as inc_v1

from tensorflow.python.keras.optimizers import Adam, RMSprop
from tensorflow.python.keras.backend import int_shape
from tensorflow.python.keras.models import Model, save_model, load_model
from tensorflow.python.keras.optimizers import Adam, RMSprop
from tensorflow.python.keras.metrics import top_k_categorical_accuracy,categorical_accuracy
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler
from tensorflow.python.keras.utils import multi_gpu_model
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import clip_ops

import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.sparsity import keras as sparsity

# Mahya
IMAGENET_PATH = '/HD1/'
TRAIN_PATH = 'Train/'
VAL_2_PATH = 'Val_2/'
META_FILE = 'dev_kit/ILSVRC2012_devkit_t12/data/meta.mat'
CONFIG_PATH = os.getcwd()
VALID_TIME_MINUTE = 5

# Pitagyro
'''IMAGENET_PATH = '/HD1/'
TRAIN_PATH = 'ILSVRC2012_img_train/'
VAL_2_PATH = 'Val_2/'
META_FILE = 'ILSVRC2012_devkit_t12/data/meta.mat'
CONFIG_PATH = os.getcwd()
VALID_TIME_MINUTE = 5'''

def main():
    model_path = './'
    model_type = 'googlenet'

    if model_type == 'resource_scalable':
        # Load ofms list from .txt file
        ofms = []
        with open(model_path + 'ofms.txt') as f:
            for line in f:
                ofm = line[:-1]
                ofms.append(ofm)
        model = rs_net_ch(num_classes=num_classes, ofms=ofms)
        model = tu.load_model_npy(model, model_path + 'weights.npy')
    elif model_type == 'googlenet':
        model = inc_v1.InceptionV1(include_top=True, weights='imagenet')
        #model.load_weights(model_path + 'googlenet_weights.h5')

    epochs = 64

    pruned_model = pu.prune_model(model, model_type='googlenet',
                                  batch_size=64,
                                  imagenet_path=IMAGENET_PATH,
                                  train_path=TRAIN_PATH,
                                  val_path=VAL_2_PATH,
                                  meta_path=META_FILE,
                                  tb_logpath=model_path+"logs",
                                  num_epochs=epochs)

    local_accuracy = eu.get_local_accuracy(pruned_model, IMAGENET_PATH,
                                           VAL_2_PATH, model_path + 'selected_dirs.txt')
    global_acc, raw_acc = eu.get_global_accuracy(pruned_model, num_classes, IMAGENET_PATH,
                                                 VAL_2_PATH, META_FILE,
                                                 model_path + 'selected_dirs.txt',
                                                 raw_acc=True)

    # Write pruned model summary to txt file
    orig_stdout = sys.stdout
    f = open('pruned_model_summary.txt', 'w')
    sys.stdout = f
    print(pruned_model.summary())
    sys.stdout = orig_stdout
    f.close()

    print("\nRaw Accuracy: " + str(raw_acc))
    print("Local Accuracy: " + str(local_accuracy))
    print("Global Accuracy: " + str(global_acc))
    print("\nWriting results to file...")
    with open(model_path + 'prune_model_accuracy.txt', 'w') as f:
        f.write('Local Accuracy: %d' % local_accuracy)
        f.write('Global Accuracy: %d' % global_acc)
        f.write('Raw Accuracy: %d' % raw_acc)

if __name__ == '__main__':
    main()
