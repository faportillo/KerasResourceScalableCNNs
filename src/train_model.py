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
import src.utils.train_utils as tu
from src.GoogLeNet.googlenet_rs import googlenet_rs
from src.MobileNet.mobilenet_rs import mobilenet_rs
from src.MobileNet.mobilenet_rs_25layer import mobilenet_rs_25layer
<<<<<<< HEAD
from src.MobileNet.mobilenet_rs_25width import mobilenet_rs_25width
=======
from src.MobileNet.mobilenet_rs_5layer import mobilenet_rs_5layer

>>>>>>> d239dc4f4cc393c57552208f717bda26c8c17f1a

import src.GoogLeNet.VanillaGoogLeNet.inception_v1 as inc_v1
from tensorflow.python.keras.applications.mobilenet import MobileNet

from tensorflow.python.keras.optimizers import Adam, RMSprop
from tensorflow.python.keras.backend import int_shape
from tensorflow.python.keras.models import Model, save_model, load_model
from tensorflow.python.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.python.keras.metrics import top_k_categorical_accuracy,categorical_accuracy
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler
from tensorflow.python.keras.utils import multi_gpu_model
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import clip_ops

import config as cfg

def train_model():
    if not os.path.exists(cfg.model_path):
        os.makedirs(cfg.model_path)

    if cfg.model_type == 'googlenet_rs':
        if path.exists(cfg.model_path + 'ofms.txt'):
            ofms = []
            with open(cfg.model_path + 'ofms.txt') as f:
                for line in f:
                    ofm = line[:-1]
                    ofms.append(int(ofm))
        else:
            if cfg.num_classes == 5:
                ofms = [13, 13, 38, 13, 19, 3, 26, 6, 6, 26, 26, 6, 38, 19, 13, 38, 19, 3, 42, 10,
                        13, 32, 22, 5, 45, 13, 13, 26, 26, 5, 51, 13, 13, 28, 36, 8, 72, 16, 16, 64,
                        40, 8, 80, 32, 32, 64, 40, 8, 80, 32, 32, 96, 48, 12, 96, 32, 32, 5]
            elif cfg.num_classes == 10:
                ofms = [13, 13, 38, 13, 19, 3, 26, 6, 6, 26, 26, 6, 38, 19, 13, 38, 19, 3, 42, 10, 13, 32, 22,
                        5, 45, 13, 13, 26, 26, 5, 51, 13, 13, 28, 36, 8, 72, 16, 16, 64, 40, 8, 80, 32, 32, 64,
                        40, 8, 80, 32, 32, 113, 56, 14, 113, 38, 38, 10]
            elif cfg.num_classes == 15:
                ofms = [15, 15, 44, 15, 22, 4, 29, 7, 7, 29, 29, 7, 44, 22, 15, 44, 22, 4, 48, 11,
                        15, 37, 26, 6, 51, 15, 15, 29, 29, 6, 59, 15, 15, 26, 33, 7, 66, 15, 15, 59,
                        37, 7, 74, 29, 29, 59, 37, 7, 74, 29, 29, 88, 44, 11, 88, 29, 29, 15]
            elif cfg.num_classes == 20:
                ofms = [13, 13, 38, 13, 19, 3, 26, 6, 6, 26, 26, 6, 38, 19, 13, 38, 19, 3, 42, 10, 13,
                        32, 22, 5, 45, 13, 13, 26, 26, 5, 51, 13, 13, 28, 36, 8, 72, 16, 16, 62, 39,
                        8, 78, 31, 31, 83, 52, 10, 103, 41, 41, 128, 64, 16, 128, 43, 43, 20]
            else:
                print("Unsupported number of classes for model type." +
                      " Currently support exists for 5, 10, 15, and 20 classes. Terminating...")
                exit(1)
            with open(cfg.model_path + '/ofms.txt', 'w') as f:
                for ofm in ofms:
                    f.write("%s\n" % ofm)

        model = googlenet_rs(num_classes=cfg.num_classes, ofms=ofms, use_aux=cfg.use_aux)

    elif cfg.model_type == 'mobilenet_rs':
        if path.exists(cfg.model_path + 'ofms.txt'):
            ofms = []
            with open(cfg.model_path + 'ofms.txt') as f:
                for line in f:
                    ofm = line[:-1]
                    ofms.append(int(ofm))
        else:
            if cfg.num_classes == 5:
                if cfg.lambda_val == 0.25:
                    ofms = [32, 43, 107, 116, 197, 197, 197, 256, 512, 512, 5]
                elif cfg.lambda_val == 0.045:
                    ofms = [16, 21, 43, 43, 85, 85, 171, 128, 256, 256, 5]
            elif cfg.num_classes == 10:
                if cfg.lambda_val == 0.25:
                    ofms = [32, 64, 116, 116, 197, 197, 213, 301, 569, 569, 10]
                elif cfg.lambda_val == 0.045:
                    ofms = [16, 32, 64, 64, 116, 122, 128, 138, 250, 250, 10]
            elif cfg.num_classes == 15:
                if cfg.lambda_val == 0.25:
                    ofms = [32, 43, 107, 116, 213, 213, 284, 320, 512, 512, 15]
                elif cfg.lambda_val == 0.045:
                    ofms = [16, 32, 64, 67, 135, 135, 138, 138, 263, 263, 15]
            elif cfg.num_classes == 20:
                if cfg.lambda_val == 0.25:
                    ofms = [32, 43, 107, 116, 213, 213, 284, 320, 512, 512, 20]
                elif cfg.lambda_val == 0.045:
                    ofms = [16, 21, 53, 58, 107, 107, 142, 160, 256, 256, 20]
            else:
                print("Unsupported number of classes for model type." +
                      " Currently support exists for 5, 10, 15, and 20 classes. Terminating...")
                exit(1)
            with open(cfg.model_path + '/ofms.txt', 'w') as f:
                for ofm in ofms:
                    f.write("%s\n" % ofm)

        model = mobilenet_rs(num_classes=cfg.num_classes, ofms=ofms)
    
    elif cfg.model_type == 'mobilenet_rs_layer':
        if path.exists(cfg.model_path + 'ofms.txt'):
            ofms = []
            with open(cfg.model_path + 'ofms.txt') as f:
                for line in f:
                    ofm = line[:-1]
                    ofms.append(int(ofm))
        else:
            if cfg.num_classes == 5:
                if cfg.lambda_val == 0.25:
                    ofms = [32, 64, 128, 128, 256, 256, 512, 512, 512, 512, 5]
                elif cfg.lambda_val == 0.045:
                    ofms = [32, 64, 128, 128, 256, 256, 512, 5]
            elif cfg.num_classes == 10:
                if cfg.lambda_val == 0.25:
                    ofms = [32, 64, 116, 116, 197, 197, 213, 301, 569, 569, 10]
                elif cfg.lambda_val == 0.045:
                    ofms = [16, 32, 64, 64, 116, 122, 128, 138, 250, 250, 10]
            elif cfg.num_classes == 15:
                if cfg.lambda_val == 0.25:
                    ofms = [32, 43, 107, 116, 213, 213, 284, 320, 512, 512, 15]
                elif cfg.lambda_val == 0.045:
                    ofms = [16, 32, 64, 67, 135, 135, 138, 138, 263, 263, 15]
            elif cfg.num_classes == 20:
                if cfg.lambda_val == 0.25:
                    ofms = [32, 43, 107, 116, 213, 213, 284, 320, 512, 512, 20]
                elif cfg.lambda_val == 0.045:
                    ofms = [16, 21, 53, 58, 107, 107, 142, 160, 256, 256, 20]
            else:
                print("Unsupported number of classes for model type." +
                      " Currently support exists for 5, 10, 15, and 20 classes. Terminating...")
                exit(1)
            with open(cfg.model_path + '/ofms.txt', 'w') as f:
                for ofm in ofms:
                    f.write("%s\n" % ofm)
        if cfg.lambda_val == 0.25:
            model = mobilenet_rs_25layer(num_classes=cfg.num_classes, ofms=ofms)
        elif cfg.lambda_val == 0.045:
            model = mobilenet_rs_5layer(num_classes=cfg.num_classes, ofms=ofms)
    
    elif cfg.model_type == 'mobilenet_rs_width':
        if path.exists(cfg.model_path + 'ofms.txt'):
            ofms = []
            with open(cfg.model_path + 'ofms.txt') as f:
                for line in f:
                    ofm = line[:-1]
                    ofms.append(int(ofm))
        else:
            if cfg.num_classes == 5:
                if cfg.lambda_val == 0.25:
                    ofms = [32, 64, 128, 128, 256, 256, 512, 512, 1024, 1024, 5]
                elif cfg.lambda_val == 0.045:
                    ofms = [32, 64, 128, 128, 256, 256, 512, 512, 1024, 1024, 5]
        model = mobilenet_rs_25width(num_classes=cfg.num_classes, ofms=ofms)

    elif cfg.model_type == 'googlenet':
        model = inc_v1.InceptionV1(include_top=True, weights='imagenet')

    elif cfg.model_type == 'mobilenet':
        model = MobileNet()

    if cfg.optimizer == 'adam':
        opt = Adam(lr=cfg.learning_rate)
    elif cfg.optimizer == 'rmsprop':
        opt = RMSprop(lr=cfg.learning_rate)
    elif cfg.optimizer == 'sgd':
        opt = SGD(lr=cfg.learning_rate)
    else:
        print("Invalid optimizer. Terminating...")
        exit(1)

    if cfg.use_aux:
        model.compile(optimizer=opt, loss=[tu.focal_loss(alpha=.25, gamma=1),
                                            tu.focal_loss(alpha=.25, gamma=1)],
                      metrics=[categorical_accuracy, tu.global_accuracy, tu.local_accuracy],
                      loss_weights=[1.0, 0.3])
    else:
        model.compile(optimizer=opt, loss=[tu.focal_loss(alpha=.25, gamma=1)],
                      metrics=[categorical_accuracy, tu.global_accuracy, tu.local_accuracy])

    if cfg.load_weights is True:
        model = tu.load_model_npy(model, 'weights.npy')

    model = tu.fit_model(model,
                         model_type=cfg.model_type,
                         num_classes=cfg.num_classes,
                         batch_size=cfg.batch_size,
                         val_batch_size=cfg.val_batch_size,
                         val_period=cfg.validation_period,
                         op_type=cfg.optimizer,
                         format=cfg.format,
                         image_size=cfg.image_size,
                         symlink_prefix=cfg.symlnk_prfx,
                         model_path=cfg.model_path,
                         imagenet_path=cfg.IMAGENET_PATH,
                         train_path=cfg.TRAIN_PATH,
                         val_path=cfg.VAL_2_PATH,
                         meta_path=cfg.META_FILE,
                         tb_logpath=cfg.model_path + "/logs",
                         num_epochs=cfg.num_epochs,
                         augment=cfg.augment_data,
                         garbage_multiplier=cfg.garbage_multiplier,
                         workers=cfg.workers,
                         max_queue_size=cfg.max_queue_size)

    shutil.move("selected_dirs.txt", cfg.model_path)