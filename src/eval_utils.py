from __future__ import print_function
import sys

sys.path.append("..")
import tensorflow as tf
import numpy as np
import train_utils as tu

import os
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

from scipy.io import loadmat

'''
    Both functions:
        get_local_accuracy
        get_global_accuracy
    iterate through entire dataset one-by-one to find their respective metrics
    
'''
def get_local_accuracy(model, imagnenet_path, val_path, selected_dirs_file):
    selected_dirs = []
    selected_dirs.append('')
    f = open(selected_dirs_file, 'r')
    for line in f:
        line = line.strip('\n')
        selected_dirs.append(line)
    f.close()

    selected_dirs = sorted(selected_dirs)
    validation_path = os.path.join(imagnenet_path, val_path)
    all_dirs = os.listdir(validation_path)
    local_acc_list = []

    total_imgs = 0
    correct_imgs = 0
    for folder in all_dirs:
        correct_index = 0
        if folder in selected_dirs:
            correct_index = selected_dirs.index(folder)
        else:
            continue

        p = os.path.join(validation_path, folder)
        all_imgs = os.listdir(p)
        for elem in all_imgs:
            file_name = os.path.join(p, elem)
            img = Image.open(file_name)
            img = img.resize((227, 227))
            img = np.array(img)
            img = img / 255.0
            if (len(img.shape) != 3) or (img.shape[0] * img.shape[1] * img.shape[2] !=
                                         (227 * 227 * 3)):
                # print ("Wrong format skipped")
                continue
            img = img.reshape(1, 227, 227, 3)
            pred = model.predict(img)
            total_imgs += 1
            if np.argmax(pred[0]) == correct_index:
                correct_imgs += 1

    print("Total: " + str(total_imgs) + ", correct: " + \
          str(correct_imgs) + ", LOCAL ACC:" + str(correct_imgs * 1.0 / total_imgs))
    local_acc = correct_imgs * 1.0 / total_imgs
    return local_acc

def get_raw_accuracy(model, num_classes, imagnenet_path, val_path, meta_file, selected_dirs_file, symlink_prefix='GARBAGE'):
    selected_dirs = []
    selected_dirs.append('')
    f = open(selected_dirs_file, 'r')
    for line in f:
        line = line.strip('\n')
        selected_dirs.append(line)
    f.close()

    val_img_path = os.path.join(imagnenet_path, val_path)
    global_validation_path = os.path.join(imagnenet_path, symlink_prefix+"_VAL_")
    wnid_labels, _ = tu.load_imagenet_meta(os.path.join(imagnenet_path, meta_file))
    selected_classes = tu.create_garbage_links(num_classes, wnid_labels, '', '',
                                               val_img_path, global_validation_path)
    tu.create_garbage_class_folder(selected_classes, wnid_labels, '', '',
                                   val_img_path, global_validation_path)

    tu.change_garbage_class_folder_val(selected_classes, wnid_labels,
                                       val_img_path, global_validation_path)

    all_dirs = os.listdir(global_validation_path)
    total_imgs = 0
    correct_imgs = 0
    garb_count = 0

    for folder in all_dirs:
        correct_index = 0
        if folder in selected_dirs:
            correct_index = selected_dirs.index(folder)
        elif folder == 'gclass':
            correct_index = 0
        else:
            continue
        print(correct_index)
        p = os.path.join(global_validation_path, folder)
        all_imgs = os.listdir(p)

        for elem in all_imgs:
            if garb_count == (50000 - (num_classes-1) * 50):
                break
            if correct_index == 0:
                garb_count += 1
            file_name = os.path.join(p, elem)
            img = Image.open(file_name)
            img = img.resize((227, 227))
            img = np.array(img)
            img = img / 255.0
            if (len(img.shape) != 3) or (img.shape[0] * img.shape[1] * img.shape[2] !=
                                         (227 * 227 * 3)):
                # print ("Wrong format skipped")
                continue
            img = img.reshape(1, 227, 227, 3)
            pred = model.predict(img)
            total_imgs += 1
            if np.argmax(pred[0]) == correct_index:
                correct_imgs += 1
            print("Total: " + str(total_imgs) + ", correct: " + \
                  str(correct_imgs) + ", GLOBAL ACC:" + str(correct_imgs * 1.0 / total_imgs))

    raw_acc = correct_imgs * 1.0 / total_imgs
    return raw_acc

def get_global_accuracy(model, num_classes, imagnenet_path, val_path, meta_file, selected_dirs_file, raw_acc=True, symlink_prefix='GARBAGE'):
    selected_dirs = []
    selected_dirs.append('')
    f = open(selected_dirs_file, 'r')
    for line in f:
        line = line.strip('\n')
        selected_dirs.append(line)
    f.close()

    val_img_path = os.path.join(imagnenet_path, val_path)
    global_validation_path = os.path.join(imagnenet_path, symlink_prefix+"_VAL_")
    wnid_labels, _ = tu.load_imagenet_meta(os.path.join(imagnenet_path, meta_file))
    selected_classes = tu.create_garbage_links(num_classes, wnid_labels, '', '',
                                               val_img_path, global_validation_path)
    tu.create_garbage_class_folder(selected_classes, wnid_labels, '', '',
                                   val_img_path, global_validation_path)

    tu.change_garbage_class_folder_val(selected_classes, wnid_labels,
                                       val_img_path, global_validation_path)

    all_dirs = os.listdir(global_validation_path)
    total_imgs = 0
    correct_global_imgs = 0
    correct_raw_imgs = 0
    garb_count = 0

    for folder in all_dirs:
        correct_index = 0
        if folder in selected_dirs:
            correct_index = selected_dirs.index(folder)
            global_index = 1
        elif folder == 'gclass':
            correct_index = 0
            global_index = 0
        else:
            continue
        print(correct_index)
        p = os.path.join(global_validation_path, folder)
        all_imgs = os.listdir(p)

        for elem in all_imgs:
            if garb_count == (50000 - (num_classes-1) * 50):
                break
            if correct_index == 0:
                garb_count += 1
            file_name = os.path.join(p, elem)
            img = Image.open(file_name)
            img = img.resize((227, 227))
            img = np.array(img)
            img = img / 255.0
            if (len(img.shape) != 3) or (img.shape[0] * img.shape[1] * img.shape[2] !=
                                         (227 * 227 * 3)):
                # print ("Wrong format skipped")
                continue
            img = img.reshape(1, 227, 227, 3)
            pred = model.predict(img)
            if isinstance(pred, list): # len(pred.shape) == 3 or len(pred.shape) == 4:
                print(np.argmax(pred[0]))
                print(np.argmax(pred[1]))
                total_imgs += 1

                if np.argmax(pred[0]) > 0:
                    pred_class = 1
                else:
                    pred_class = 0
                if pred_class == global_index:
                    correct_global_imgs += 1

                if raw_acc and (np.argmax(pred[0]) == correct_index):
                    correct_raw_imgs += 1
            elif not isinstance(pred, list) and len(pred.shape) == 2:
                print(np.argmax(pred))
                total_imgs += 1

                if np.argmax(pred) > 0:
                    pred_class = 1
                else:
                    pred_class = 0
                if pred_class == global_index:
                    correct_global_imgs += 1

                if raw_acc and (np.argmax(pred) == correct_index):
                    correct_raw_imgs += 1
            else:
                print("Invalid prediction shape...")

            print("[Raw] Total: " + str(total_imgs) + ", correct: " + \
                  str(correct_raw_imgs) + ", GLOBAL ACC:" + str(correct_raw_imgs * 1.0 / total_imgs))
            print("[Global] Total: " + str(total_imgs) + ", correct: " + \
                  str(correct_global_imgs) + ", GLOBAL ACC:" + str(correct_global_imgs * 1.0 / total_imgs))
            print()
    global_acc = correct_global_imgs * 1.0 / total_imgs
    if raw_acc:
        raw_acc = correct_raw_imgs * 1.0 / total_imgs
        return global_acc, raw_acc
    else:
        return global_acc

