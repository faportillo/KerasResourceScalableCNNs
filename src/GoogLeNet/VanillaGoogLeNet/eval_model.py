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
    model_path = './L20_s3_trial2_pg/'
    model_type = 'googlenet'

    is_pruned = False

    if model_type == 'googlenet':
        model = inc_v1.InceptionV1(include_top=True, weights='imagenet')
        #model.load_weights(model_path + 'googlenet_weights.h5')
    elif model_type == 'mobilenet':
        return  # put mobilenet code here
    else:  # some form of resource-scalable cnn
        # Load ofms list from .txt file
        ofms = []
        with open(model_path + 'ofms.txt') as f:
            for line in f:
                ofm = line[:-1]
                ofms.append(ofm)
        num_classes = int(ofms[-1])  # number of classes

        # Create model
        model = rs_net_ch(num_classes=num_classes, ofms=ofms)
        model = tu.load_model_npy(model, model_path + 'max_l_g_weights.npy')

    '''model.compile(optimizer='adam', loss=[tu.focal_loss(alpha=.25, gamma=2)],
                  metrics=[categorical_accuracy, tu.global_accuracy, tu.local_accuracy])'''
    if is_pruned:
        import src.prune_utils as pu
        sparsity_val = pu.calculate_sparsity(model)
        print('\n\nCalculating sparsity... ')
        print(sparsity_val)
        print('\n\n')
        with open(model_path + 'sparsity_pruning_logs.txt', 'a+') as f:
            f.write('\nFINAL SPARSITY: %f\n' % sparsity_val)
    # Write pruned model summary to txt file
    orig_stdout = sys.stdout
    f = open(model_path + 'model_summary.txt', 'w')
    sys.stdout = f
    print(model.summary())
    sys.stdout = orig_stdout
    f.close()

    #Get raw, local, and global accuracy
    local_accuracy = eu.get_local_accuracy(model,
                                           IMAGENET_PATH,
                                           VAL_2_PATH,
                                           model_path + 'selected_dirs.txt',
                                           image_size=224,
                                           is_rs_model=False)
    '''global_acc, raw_acc = eu.get_global_accuracy(model, num_classes, IMAGENET_PATH,
                                                 VAL_2_PATH, META_FILE,
                                                 model_path + 'selected_dirs.txt',
                                                 raw_acc=True, symlink_prefix='_GARBAGE')'''
    global_acc = 0.0
    raw_acc = 0.0

    print("\nRaw Accuracy: " + str(raw_acc))
    print("Local Accuracy: " + str(local_accuracy))
    print("Global Accuracy: " + str(global_acc))
    print("\nWriting results to file...")
    with open(model_path + 'model_accuracy.txt', 'w') as f:
        f.write('Machine: pitagyro\n')
        f.write(model_path + '\n')
        f.write('Local Accuracy: %f\n' % local_accuracy)
        f.write('Global Accuracy: %f\n' % global_acc)
        f.write('Raw Accuracy: %f\n' % raw_acc)

if __name__ == '__main__':
    main()
