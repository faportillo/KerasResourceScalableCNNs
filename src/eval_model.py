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
#import src.utils.prune_utils as pu
import src.utils.eval_utils as eu
from src.GoogLeNet.googlenet_rs import googlenet_rs
from src.MobileNet.mobilenet_rs import mobilenet_rs
import src.GoogLeNet.VanillaGoogLeNet.inception_v1 as inc_v1
from tensorflow.python.keras.applications.mobilenet import MobileNet
from src.MobileNet.mobilenet_rs_25layer import mobilenet_rs_25layer


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

def eval_model():
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
            print("ERROR. Pruning this model type" +
                  " requires an ofms.txt file. Make sure to train model first")
            exit(1)

        model = googlenet_rs(num_classes=cfg.num_classes, ofms=ofms, use_aux=cfg.use_aux)
        model = tu.load_model_npy(model, cfg.model_path + cfg.eval_weight_file)

    elif cfg.model_type == 'mobilenet_rs':
        if path.exists(cfg.model_path + 'ofms.txt'):
            ofms = []
            with open(cfg.model_path + 'ofms.txt') as f:
                for line in f:
                    ofm = line[:-1]
                    ofms.append(int(ofm))
        else:
            print("ERROR. Pruning this model type " +
                  "requires an ofms.txt file. Make sure to train model first")
            exit(1)
        model = mobilenet_rs(num_classes=cfg.num_classes, ofms=ofms)
        model = tu.load_model_npy(model, cfg.model_path + cfg.eval_weight_file)
    
    elif cfg.model_type == 'mobilenet_rs_layer':
        if path.exists(cfg.model_path + 'ofms.txt'):
            ofms = []
            with open(cfg.model_path + 'ofms.txt') as f:
                for line in f:
                    ofm = line[:-1]
                    ofms.append(int(ofm))
        else:
            print("ERROR. Pruning this model type " +
                  "requires an ofms.txt file. Make sure to train model first")
            exit(1)
        model = mobilenet_rs_25layer(num_classes=cfg.num_classes, ofms=ofms)
        if '.npy' in cfg.eval_weight_file:
            model = tu.load_model_npy(model, cfg.model_path + cfg.eval_weight_file)
        elif '.h5' in cfg.eval_weight_file:
            model = load_model(cfg.model_path + cfg.eval_weight_file)
    
    elif cfg.model_type == 'googlenet':
        model = inc_v1.InceptionV1(include_top=True, weights='imagenet')
        if cfg.is_pruned:
            model = load_model(cfg.model_path + 'final_pruned_model.h5')
    elif cfg.model_type == 'mobilenet':
        model = MobileNet()
        if cfg.is_pruned:
            model = load_model(cfg.model_path + 'final_pruned_model.h5')
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

    if cfg.is_pruned:
        import src.utils.prune_utils as pu
        sparsity_val = pu.calculate_sparsity(model)
        print('\n\nCalculating sparsity... ')
        print(sparsity_val)
        print('\n\n')
        if cfg.is_finetuned:
            with open(cfg.model_path + 'ft_sparsity_pruning_logs.txt', 'a+') as f:
                f.write('\nFINAL SPARSITY: %f\n' % sparsity_val)
        else:
            with open(cfg.model_path + 'sparsity_pruning_logs.txt', 'a+') as f:
                f.write('\nFINAL SPARSITY: %f\n' % sparsity_val)

    if '_rs' in cfg.model_type:
        # Get Local and Global Accuracy for Resource-Scalable Models
        print("GETTING LOCAL ACCURACY")
        local_accuracy = eu.get_local_accuracy(model, cfg.IMAGENET_PATH,
                                               cfg.VAL_2_PATH, cfg.model_path + 'selected_dirs.txt')
        print("GETTING GLOBAL ACCURACY")
        global_acc, raw_acc = eu.get_global_accuracy(model, cfg.num_classes, cfg.IMAGENET_PATH,
                                                     cfg.VAL_2_PATH, cfg.META_FILE,
                                                     cfg.model_path + 'selected_dirs.txt',
                                                     raw_acc=True,
                                                     symlink_prefix=cfg.symlnk_prfx)

    else:
        '''
            If using vanilla model, may just want to do regular categorical accuracy
            
            Unless model_path contains a 'selected_dirs.txt' file,
            then can do local & global accuracy
        '''
        if path.exists(cfg.model_path + 'selected_dirs.txt'):
            print("GETTING LOCAL ACCURACY")
            local_accuracy = eu.get_local_accuracy(model,
                                                   cfg.IMAGENET_PATH,
                                                   cfg.VAL_2_PATH,
                                                   cfg.model_path + 'selected_dirs.txt',
                                                   image_size=cfg.image_size,
                                                   is_rs_model=False)
            print("GETTING GLOBAL ACCURACY")
            global_acc, raw_acc = eu.get_global_accuracy(model,
                                                       cfg.num_classes,
                                                       cfg.IMAGENET_PATH,
                                                       cfg.VAL_2_PATH,
                                                       cfg.META_FILE,
                                                       raw_acc=True,
                                                       selected_dirs_file=cfg.model_path + 'selected_dirs.txt',
                                                       image_size=cfg.image_size,
                                                       symlink_prefix=cfg.symlnk_prfx,
                                                       is_rs_model=False)

        else:
            local_accuracy = 0.0
            print("GETTING GLOBAL ACCURACY")
            global_acc, raw_acc = eu.get_global_accuracy(model,
                                                       cfg.num_classes,
                                                       cfg.IMAGENET_PATH,
                                                       cfg.VAL_2_PATH,
                                                       cfg.META_FILE,
                                                       raw_acc=True,
                                                       selected_dirs_file=None,
                                                       image_size=cfg.image_size,
                                                       symlink_prefix=cfg.symlnk_prfx,
                                                       is_rs_model=False)

    print("\nRaw Accuracy: " + str(raw_acc))
    print("Local Accuracy: " + str(local_accuracy))
    print("Global Accuracy: " + str(global_acc))
    print("\nWriting results to file...")
    if cfg.is_pruned:
        if cfg.is_finetuned:
            with open(cfg.model_path + 'ft_pruned_model_accuracy.txt', 'w') as f:
                f.write('Machine:' + cfg.machine_name + '\n')
                f.write(cfg.model_path + '\n')
                f.write('Local Accuracy: %f\n' % local_accuracy)
                f.write('Global Accuracy: %f\n' % global_acc)
                f.write('Raw Accuracy: %f\n' % raw_acc)
        else:
            with open(cfg.model_path + 'pruned_model_accuracy.txt', 'w') as f:
                f.write('Machine:' + cfg.machine_name + '\n')
                f.write(cfg.model_path + '\n')
                f.write('Local Accuracy: %f\n' % local_accuracy)
                f.write('Global Accuracy: %f\n' % global_acc)
                f.write('Raw Accuracy: %f\n' % raw_acc)
    else:
        if cfg.is_finetuned:
            with open(cfg.model_path + 'ft_model_accuracy.txt', 'w') as f:
                f.write('Machine:' + cfg.machine_name + '\n')
                f.write(cfg.model_path + '\n')
                f.write('Local Accuracy: %f\n' % local_accuracy)
                f.write('Global Accuracy: %f\n' % global_acc)
                f.write('Raw Accuracy: %f\n' % raw_acc)
        else:
            with open(cfg.model_path + 'model_accuracy.txt', 'w') as f:
                f.write('Machine:' + cfg.machine_name + '\n')
                f.write(cfg.model_path + '\n')
                f.write('Local Accuracy: %f\n' % local_accuracy)
                f.write('Global Accuracy: %f\n' % global_acc)
                f.write('Raw Accuracy: %f\n' % raw_acc)


if __name__ == '__main__':
    main()
