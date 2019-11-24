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
IMAGENET_PATH = '/HD1/'
TRAIN_PATH = 'ILSVRC2012_img_train/'
VAL_2_PATH = 'Val_2/'
META_FILE = 'ILSVRC2012_devkit_t12/data/meta.mat'
CONFIG_PATH = os.getcwd()
VALID_TIME_MINUTE = 5

# MC
'''IMAGENET_PATH = '/HD1/'
TRAIN_PATH = 'train/'
VAL_2_PATH = 'val/'
META_FILE = 'ILSVRC2012_devkit_t12/data/meta.mat'
CONFIG_PATH = os.getcwd()
VALID_TIME_MINUTE = 5'''

def main():

    model_path = './L20_s3_trial3/'

    is_pruned = True
    first_class = 0
    last_class = 49
    num_epochs = 200
    b_size = 64  # Batch size
    val_b_size = 64
    validation_period = 10
    learning_rate = 0.0001
    augment_data = True  # Augment data or not
    load_weights = False
    use_tfrecord_format = False
    use_aux = True
    num_outs=2
    format = 'generator'
    symlnk_prfx = '4GARBAGE'

    # Load ofms list from .txt file
    ofms = []
    with open(model_path + 'ofms.txt') as f:
        for line in f:
            ofm = line[:-1]
            ofms.append(int(ofm))

    num_classes = ofms[-1]  # number of classes

    # Create model
    model = rs_net_ch(num_classes=num_classes, ofms=ofms)
    
    if is_pruned:
        import src.prune_utils as pu
        model = load_model(model_path + 'pruned_max_l_g_weights.h5')
        sparsity_val = pu.calculate_sparsity(model)
        print('\n\nCalculating sparsity... ')
        print(sparsity_val)
        print('\n\n')
        with open(model_path + 'sparsity_pruning_logs.txt', 'a+') as f:
            f.write('\nFINAL SPARSITY: %f\n' % sparsity_val)
        
    else:
        model = tu.load_model_npy(model, model_path + 'max_l_g_weights.npy')
    # Write pruned model summary to txt file
    orig_stdout = sys.stdout
    f = open(model_path + 'model_summary.txt', 'w')

    sys.stdout = f
    print(model.summary())
    sys.stdout = orig_stdout
    f.close()
    

    adam = Adam(lr=learning_rate)
    # model.compile(optimizer=adam, loss=[focal_loss(alpha=.25, gamma=2),
    # focal_loss(alpha=.25, gamma=2)], metrics=[categorical_accuracy],loss_weights=[1.0,0.3])
    if use_aux:
        model.compile(optimizer=adam, loss=[tu.focal_loss(alpha=.25, gamma=1),
                                            tu.focal_loss(alpha=.25, gamma=1)],
                      metrics=[categorical_accuracy, tu.global_accuracy, tu.local_accuracy],
                      loss_weights=[1.0, 0.3])
    else:
        model.compile(optimizer=adam, loss=[tu.focal_loss(alpha=.25, gamma=1)],
                      metrics=[categorical_accuracy, tu.global_accuracy, tu.local_accuracy])


    # Train model
    model_ft = tu.fit_model(model,
                                 num_classes,
                                 first_class,
                                 last_class,
                                 batch_size=b_size,
                                 val_batch_size=val_b_size,
                                 val_period=validation_period,
                                 op_type='adam',
                                 format=format,
                                 symlink_prefix=symlnk_prfx,
                                 model_path=model_path,
                                 imagenet_path=IMAGENET_PATH,
                                 train_path=TRAIN_PATH,
                                 val_path=VAL_2_PATH,
                                 meta_path=META_FILE,
                                 tb_logpath=model_path + "/logs",
                                 num_epochs=num_epochs,
                                 augment=augment_data,
                                 garbage_multiplier=8,
                                 num_outs=num_outs,
                                 workers=4,
                                 finetuning=True)

    shutil.move("selected_dirs.txt", model_path)


if __name__ == '__main__':
    main()
