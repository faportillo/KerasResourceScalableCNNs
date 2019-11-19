import os
import random
import tensorflow as tf
import numpy as np
import math
from scipy.io import loadmat
from PIL import Image
import argparse, shutil

# from Data_Augmentation import augment
from tensorflow.python.keras.layers import Input, Dense, Convolution2D, \
    MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, Reshape, \
    Activation, Concatenate, Layer, Lambda
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.backend import int_shape
from tensorflow.python.keras.models import Model, save_model, load_model
from tensorflow.python.keras.optimizers import Adam, RMSprop
from tensorflow.python.keras.metrics import top_k_categorical_accuracy, \
    categorical_accuracy
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, \
    Callback, LearningRateScheduler, TerminateOnNaN
from tensorflow.python.ops import clip_ops

# from model_shell import google_csn

IMAGE_SIZE = 227
ROT_RANGE = 30
WIDTH_SHIFT_RANGE = 0.2
HEIGHT_SHIFT_RANGE = 0.2
ZOOM_RANGE = 0.2
SHEAR_RANGE = 0.2
HORIZONTAL_FLIP = True
VALIDATION_BATCH_SIZE = 32

def focal_loss(gamma=2., alpha=.25):
  def mtmd_nl(target, output, from_logits=False, axis=-1):
    rank = len(output.shape)
    axis = axis % rank
    output = output / K.sum(output, axis, True)
    #epsilon_ = K._to_tensor(epsilon(), output.dtype.base_dtype)
    epsilon_ = tf.ones_like(output) * K.epsilon();
    output = clip_ops.clip_by_value(output, epsilon_, 1. - epsilon_)
    return -K.sum(target * K.pow((1 - output), gamma) * K.log(output), axis)
  return mtmd_nl

def dual_loss(gamma=2, alpha=.25):
    '''
        Dual Loss = Loss_Global + Loss_Local
                  = Focal_Loss + (y==1)*Cross_Entropy_Loss
    '''
    def mtmd_dl(target, output, axis=-1):
        rank = len(output.shape)
        axis = axis % rank
        output = output / K.sum(output, axis, True)
        # epsilon_ = K._to_tensor(epsilon(), output.dtype.base_dtype)
        epsilon_ = tf.ones_like(output) * K.epsilon();
        output = clip_ops.clip_by_value(output, epsilon_, 1. - epsilon_)
        global_loss = -K.sum(target * K.pow((1 - output), gamma) * K.log(output), axis)
        argmax_true = K.argmax(y_true, axis=-1)
        where_true = K.not_equal(argmax_true, zero)
        local_loss = where_true * K.categorical_crossentropy(target, output)
        return global_loss + local_loss

    return mtmd_dl

def global_accuracy(y_true, y_pred):
    '''
    Assumes garbage class has a '0' label
    Checks if model output belongs to the desired scope of interest or not
    :param y_true: Model output, list of output tensors
    :param y_pred: Dataset labels, list of ints in a batch
    :return: global accuracy
    '''
    argmax_true = K.argmax(y_true, axis=-1)
    argmax_pred = K.argmax(y_pred, axis=-1)
    argmax_true = K.cast(argmax_true, dtype='float32')
    argmax_pred = K.cast(argmax_pred, dtype='float32')
    zero = K.constant(0, dtype='float32')
    where_true = K.not_equal(argmax_true, zero)
    where_pred = K.not_equal(argmax_pred, zero)
    return K.cast(K.equal(where_true, where_pred), K.floatx())

def local_accuracy(y_true, y_pred):
    y_true = K.cast(y_true, dtype='float32')
    y_pred = K.cast(y_pred, dtype='float32')
    zero = K.constant(0, dtype='float32')
    argmax_true = K.argmax(y_true, axis=-1)
    argmax_pred = K.argmax(y_pred, axis=-1)
    argmax_true = K.cast(argmax_true, dtype='float32')
    argmax_pred = K.cast(argmax_pred, dtype='float32')
    non_zero_true = K.gather(argmax_true, tf.where(K.not_equal(argmax_true, zero)))
    non_zero_pred = K.gather(argmax_pred, tf.where(K.not_equal(argmax_true, zero)))
    return K.cast(K.equal(non_zero_true, non_zero_pred), K.floatx())

def imagenet_generator(train_data_path,
                       val_data_path,
                       batch_size,
                       do_augment,
                       val_batch_size=VALIDATION_BATCH_SIZE,
                       image_size=IMAGE_SIZE):
    '''
        train_data_path: Path for ImageNet Training Directory
        val_data_path: Path for ImageNet Validation Directory
        :return: Keras Data Generators for Training and Validation
    '''
    if do_augment == True:
        rot_range = ROT_RANGE
        w_shift_r = WIDTH_SHIFT_RANGE
        h_shift_r = HEIGHT_SHIFT_RANGE
        z_range = ZOOM_RANGE
        shear_r = SHEAR_RANGE
        h_flip = True
    else:
        rot_range = 0
        w_shift_r = 0.0
        h_shift_r = 0.0
        z_range = 0.0
        shear_r = 0.0
        h_flip = False

    print("Grabbing Training Dataset")
    train_datagen = ImageDataGenerator(samplewise_center=False, \
                                       rotation_range=rot_range, \
                                       width_shift_range=w_shift_r, \
                                       height_shift_range=h_shift_r, \
                                       zoom_range=z_range, \
                                       shear_range=shear_r, \
                                       horizontal_flip=h_flip, \
                                       fill_mode='nearest', rescale=1. / 255)

    val_datagen = ImageDataGenerator(rescale=1. / 255)

    '''
      Change follow_links to True when using symbolic links
      to training and validation data
    '''
    train_generator = train_datagen.flow_from_directory(train_data_path,
                                                        target_size=(image_size, image_size),
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        class_mode='categorical',
                                                        follow_links=True)
    print("Grabbing Validation Dataset")
    validation_generator = val_datagen.flow_from_directory(val_data_path,
                                                           target_size=(image_size, image_size),
                                                           batch_size=val_batch_size,
                                                           shuffle=True,
                                                           class_mode='categorical',
                                                           follow_links=True)

    return train_generator, validation_generator

def imagenet_generator_multi(train_data_path,
                             val_data_path,
                             batch_size,
                             do_augment,
                             val_batch_size=VALIDATION_BATCH_SIZE,
                             image_size=IMAGE_SIZE,
                             num_outputs=2):
    '''
        For use with auxiliary classifiers or mutliple outputs
        train_data_path: Path for ImageNet Training Directory
        val_data_path: Path for ImageNet Validation Directory
        :return: Keras Data Generators for Training and Validation
    '''
    if do_augment == True:
        rot_range = ROT_RANGE
        w_shift_r = WIDTH_SHIFT_RANGE
        h_shift_r = HEIGHT_SHIFT_RANGE
        z_range = ZOOM_RANGE
        shear_r = SHEAR_RANGE
        h_flip = True
    else:
        rot_range = 0
        w_shift_r = 0.0
        h_shift_r = 0.0
        z_range = 0.0
        shear_r = 0.0
        h_flip = False

    # print("Grabbing Training Dataset")
    train_datagen = ImageDataGenerator(samplewise_center=False, \
                                       rotation_range=rot_range, \
                                       width_shift_range=w_shift_r, \
                                       height_shift_range=h_shift_r, \
                                       zoom_range=z_range, \
                                       shear_range=shear_r, \
                                       horizontal_flip=h_flip, \
                                       fill_mode='nearest', rescale=1. / 255)

    val_datagen = ImageDataGenerator(rescale=1. / 255)

    '''
      Change follow_links to True when using symbolic links
      to training and validation data
    '''
    train_generator = train_datagen.flow_from_directory( \
        train_data_path, target_size=(image_size, image_size), \
        batch_size=batch_size, shuffle=True, class_mode='categorical', \
        follow_links=True)
    # print("Grabbing Validation Dataset")
    validation_generator = val_datagen.flow_from_directory( \
        val_data_path, target_size=(image_size, image_size), \
        batch_size=val_batch_size, shuffle=True, \
        class_mode='categorical', \
        follow_links=True)
    if num_outputs == 2:
        multi_train_generator = create_multi_generator(train_generator)
        multi_validation_generator = create_multi_generator(validation_generator)
    else:
        multi_train_generator = create_multi_generator_3(train_generator)
        multi_validation_generator = create_multi_generator_3(validation_generator)
    return multi_train_generator, multi_validation_generator

def create_multi_generator(data_generator, batch_size=64):
    while (True):
        data_imgs, data_l = next(data_generator)
        #print(len(data_imgs))
        yield [data_imgs], [data_l, data_l]

def create_multi_generator_3(data_generator, batch_size=64):
    while (True):
        data_imgs, data_l = next(data_generator)
        yield [data_imgs], [data_l, data_l, data_l]

def create_dataset(train_data_path,
                   val_data_path,
                   batch_size,
                   do_augment=True,
                   format='generator',
                   tf_record_dir=None,
                   val_batch_size=VALIDATION_BATCH_SIZE,
                   image_size=IMAGE_SIZE,
                   num_outputs=1):
    '''
        ToDO: Replace imagenet_generator(_multi) with this function
        Uses ImageDataGenerator to create dataset from directory and either returns generators for
        train/val or converts to TFRecord file
        :return: Keras Data Generators (or TFRecord Files) for Training and Validation
    '''
    if do_augment == True:
        if format == 'tfrecord':
            raise Exception("Cannot augment images beforehand if coverting to TFRecord.")
        rot_range = ROT_RANGE
        w_shift_r = WIDTH_SHIFT_RANGE
        h_shift_r = HEIGHT_SHIFT_RANGE
        z_range = ZOOM_RANGE
        shear_r = SHEAR_RANGE
        h_flip = True
    else:
        rot_range = 0
        w_shift_r = 0.0
        h_shift_r = 0.0
        z_range = 0.0
        shear_r = 0.0
        h_flip = False

    # print("Grabbing Training Dataset")
    train_datagen = ImageDataGenerator(samplewise_center=False,
                                       rotation_range=rot_range,
                                       width_shift_range=w_shift_r,
                                       height_shift_range=h_shift_r,
                                       zoom_range=z_range,
                                       shear_range=shear_r,
                                       horizontal_flip=h_flip,
                                       fill_mode='nearest', rescale=1. / 255)

    val_datagen = ImageDataGenerator(rescale=1. / 255)

    '''
      Change follow_links to True when using symbolic links
      to training and validation data
    '''
    train_generator = train_datagen.flow_from_directory(train_data_path,
                                                        target_size=(image_size, image_size),
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        class_mode='categorical',
                                                        follow_links=True)
    # print("Grabbing Validation Dataset")
    validation_generator = val_datagen.flow_from_directory(val_data_path,
                                                           target_size=(image_size, image_size),
                                                           batch_size=val_batch_size,
                                                           shuffle=True,
                                                           class_mode='categorical',
                                                           follow_links=True)
    if num_outputs == 1:
        print("")
    elif num_outputs == 2:
        train_generator = create_multi_generator(train_generator)
        validation_generator = create_multi_generator(validation_generator)
    elif num_outputs == 3:
        train_generator = create_multi_generator_3(train_generator)
        validation_generator = create_multi_generator_3(validation_generator)
    else:
        raise Exception("Invalid num_outputs type")


    if format=='generator':
        return train_generator, validation_generator
    elif format=='tfrecord':
        if tf_record_dir is None:
            raise Exception("No tf_record_dir specified. Need to define in order to save TFRecord files in a location")
        print("Grabbing data from directories...")
        train_dataset = tf.data.Dataset.from_generator(lambda: train_generator, (tf.float32, tf.int64))
        val_dataset = tf.data.Dataset.from_generator(lambda: validation_generator, (tf.float32, tf.int64))

        def tr_generator():
            for features in train_dataset:
                print(features)
                yield serialize(*features)
        def va_generator():
            for features in val_dataset:
                print(features)
                yield  serialize(*features)
        print("Serializing datasets...")
        serialized_train_dataset = tf.data.Dataset.from_generator(tr_generator,
                                                                  output_types=tf.string,
                                                                  output_shapes=())
        serialized_val_dataset = tf.data.Dataset.from_generator(va_generator,
                                                                  output_types=tf.string,
                                                                  output_shapes=())
        print(serialized_train_dataset)
        print("Writing to .tfrecord files...")
        writer = tf.python_io.TFRecordWriter(tf_record_dir + 'train.tfrecord')
        writer.write(serialized_train_dataset)
        print("Train TFRecord File written to " + tf_record_dir + 'train.tfrecord')
        writer = tf.python_io.TFRecordWriter(tf_record_dir + 'val.tfrecord')
        writer.write(serialized_val_dataset)
        print("Validation TFRecord File written to " + tf_record_dir + 'val.tfrecord')
        return #TFRecord files
    else:
        raise Exception("Invalid format type")

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def tf_serialize(img, label, num_outputs=1):
    tf_string = tf.py_function(serialize,
                               (img, label, num_outputs),
                               (tf.string))
    return tf.reshape(tf_string, ())

def serialize(img, label, num_outputs=1):
    '''
    Used to serialize input image dataset for use in TFRecords
    :param img:
    :param label:
    :param num_outputs:
    :return:
    '''
    if num_outputs == 1:
        data = {
            'img': _float_feature(img),
            'label': _int64_feature(label),
        }
    elif num_outputs == 2:
        data = {
            'img': _float_feature(img),
            'main_label': _int64_feature(label),
            'aux_label': _int64_feature(label),
        }
    elif num_outputs == 3:
        data = {
            'img': _float_feature(img),
            'main_label': _int64_feature(label),
            'aux_label1': _int64_feature(label),
            'aux_label2': _int64_feature(label),
        }
    else:
        raise Exception("Invalid number of outputs for network. Cannot serialize data.")
    example_proto = tf.train.Example(features=tf.train.Features(feature=data))
    return example_proto.SerializeToString()

def fit_model(model, num_classes, first_class, last_class, batch_size,
              val_batch_size=50,
              val_period=1,
              image_size=227,
              op_type=None,
              decay_params=None,
              format='generator',
              imagenet_path=None,
              model_path='./',
              train_path=None,
              val_path=None,
              tb_logpath='./logs',
              meta_path=None,
              symlink_prefix='GARBAGE',
              num_epochs=1000,
              augment=True,
              clrcm_params=None,
              train_by_branch=False,
              num_outs=1,
              garbage_multiplier=1,
              workers=1):
    '''
        :param model: Keras model
        :param num_classes:
        :param batch_size:
        :param op_type: Optimizer type
        :param decay_params: Decay parameters for rmsprop
        :param imagenet_path:
        :param train_path:
        :param val_path:
        :param tb_logpath: Tensorboard Path
        :param meta_path: ImageNet meta path
        :param config_path: Config file path
        :param num_epochs:
        :param augment: Augment data (t/f)
        :param multi_outputs: Use aux classifier
        :param clrcm_params: CLRC(Cyclical Learning Rate, Cyclical Momentum for sgd
        :return:
    '''

    if num_outs > 1:
        multi_outputs = True
    else:
        multi_outputs = False

    '''
        Create dataset and link paths if using ImageDataGenerator for data
    '''
    if format == 'generator':
        orig_train_img_path = os.path.join(imagenet_path, train_path)
        orig_val_img_path = os.path.join(imagenet_path, val_path)
        train_img_path = orig_train_img_path
        val_img_path = orig_val_img_path
        wnid_labels, _ = load_imagenet_meta(os.path.join(imagenet_path, \
                                                         meta_path))
        new_training_path = os.path.join(imagenet_path, symlink_prefix+"_TRAIN_")
        new_validation_path = os.path.join(imagenet_path, symlink_prefix+"_VAL_")
        selected_classes = create_garbage_links(num_classes, wnid_labels, train_img_path,
                                                new_training_path, val_img_path, new_validation_path)
        create_garbage_class_folder(selected_classes,
                                    wnid_labels,
                                    train_img_path,
                                    new_training_path,
                                    val_img_path,
                                    new_validation_path,
                                    garbage_multiplier=garbage_multiplier)
        train_img_path = new_training_path
        val_img_path = new_validation_path

        train_data, val_data = create_dataset(train_img_path,
                                              val_img_path,
                                              batch_size=batch_size,
                                              val_batch_size=val_batch_size,
                                              do_augment=augment,
                                              num_outputs=num_outs,
                                              image_size=image_size)
    elif format == 'tfrecord':
        # Put data creation for tfrecord format
        return
    else:
        print("Invalid or unimplemented data format. Exiting...")
        exit()

    '''if multi_outputs is True:
        if num_outs == 2:
            train_data, val_data = imagenet_generator_multi(train_img_path, \
                                                            val_img_path, batch_size=batch_size,
                                                            val_batch_size=val_batch_size,\
                                                            do_augment=augment, image_size=image_size)
        else:
            train_data, val_data = imagenet_generator_multi(train_img_path, \
                                                            val_img_path, batch_size=batch_size, val_batch_size=val_batch_size, \
                                                            do_augment=augment, num_outputs=num_outs, image_size=image_size)

    else:
        train_data, val_data = imagenet_generator(train_img_path, val_img_path, \
                                                  batch_size=batch_size, val_batch_size=val_batch_size, \
                                                  do_augment=augment, image_size=image_size)'''

    '''
        Implement Callbacks
    '''
    print("Initializing Callbacks")
    tb_callback = TensorBoard(log_dir=tb_logpath)
    '''
    checkpoint_callback = ModelCheckpoint(filepath='weights.h5'\
                        ,verbose = 1, save_weights_only = True, period=1)
    '''
    termNaN_callback = TerminateOnNaN()
    save_weights_std_callback = ModelCheckpoint(model_path+'weights.hdf5',
                                                monitor='val_categorical_accuracy',
                                                verbose=1,
                                                save_best_only=True,
                                                save_weights_only=False,
                                                mode='max',
                                                period=val_period)
    callback_list = [tb_callback, termNaN_callback, save_weights_std_callback]

    '''
        If the training each branch individually, increase the number of epochs
        to be num_classes*num_epochs 
    '''
    if train_by_branch == True:
        each_branch_callback = TrainByBranch(num_classes, num_epochs)
        num_epochs *= num_classes

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
        one_cycle = OneCycle(clrcm_params[0], clrcm_params[1], clrcm_params[2],
                             clrcm_params[3], clrcm_params[4], clrcm_params[5])
        callback_list.append(one_cycle)
    else:
        print ('Invalid Optimizer. Exiting...')
        exit()

    save_weights_callback = SaveWeightsNumpy(num_classes, model,
                                             model_path,
                                             period=val_period,
                                             selected_classes=selected_classes,
                                             wnid_labels=wnid_labels,
                                             orig_train_img_path=orig_train_img_path,
                                             new_training_path=new_training_path,
                                             orig_val_img_path=orig_val_img_path,
                                             new_val_path=val_img_path,
                                             multi_outputs=multi_outputs)
    callback_list.append(save_weights_callback)

    # Fit and validate model based on generators
    if multi_outputs:
        use_multiproc = True
    else:
        use_multiproc = False

    model.fit_generator(train_data,
                        epochs=num_epochs,
                        steps_per_epoch=int(((num_classes-1) * 1300)+(1300*garbage_multiplier)) / batch_size,
                        validation_data=val_data,
                        validation_steps= int(50000 / val_batch_size),
                        validation_freq=val_period,
                        max_queue_size = 31,
                        verbose=1,
                        callbacks=callback_list,
                        workers=workers,
                        use_multiprocessing=use_multiproc)

    save_model(model, model_path+'rs_model_final.h5')

    return model

def load_model_npy(model, filename):
    print("Loading weights from: " + str(filename))
    weights = np.load(filename, encoding="latin1", allow_pickle=True)

    model.set_weights(weights)
    print("WEIGHTS LOADED")
    return model


def create_selective_symbolic_link(first_class,
                                   last_class,
                                   wnid_labels,
                                   original_training_path,
                                   new_training_path,
                                   original_validation_path,
                                   new_validation_path):

    if os.path.exists(new_training_path):
        shutil.rmtree(new_training_path)
    os.makedirs(new_training_path)
    if os.path.exists(new_validation_path):
        shutil.rmtree(new_validation_path)
    os.makedirs(new_validation_path)
    class_list = wnid_labels[first_class: last_class + 1]
    for dir in class_list:
        src = os.path.join(original_training_path, dir.strip('\n'))
        dst = os.path.join(new_training_path, dir.strip('\n'))
        os.symlink(src, dst)
        src = os.path.join(original_validation_path, dir.strip('\n'))
        dst = os.path.join(new_validation_path, dir.strip('\n'))
        os.symlink(src, dst)

def change_garbage_class_folder(selected_classes,
                                wnid_labels,
                                original_training_path,
                                new_training_path, garbage_multiplier=1):
    class_list = []
    for i in range(0, 1000):
        folder = wnid_labels[i]
        if folder in selected_classes:
            continue
        class_list.append(folder)
    train_dst = os.path.join(new_training_path, 'gclass')

    cnt = 0
    chosen_elems = []
    # number of images from garbage classes = number of images from selected classes
    while (cnt <= 1300*garbage_multiplier):
        cls_num = random.randint(0, len(class_list) - 1)
        elem = class_list[cls_num]
        train_src = os.path.join(original_training_path, elem.strip('\n'))
        train_images = os.listdir(train_src)
        img_num = random.randint(0, len(train_images) - 1)
        img = train_images[img_num]
        src_img = os.path.join(train_src, img)
        dst_img = os.path.join(train_dst, img)
        if src_img in chosen_elems:
            continue
        else:
            cnt += 1
            dst_img = os.path.join(train_dst, str(cnt) + '.JPEG')
            if os.path.isfile(dst_img):
                os.remove(dst_img)
            os.symlink(src_img, dst_img)
            chosen_elems.append(src_img)


def change_garbage_class_folder_val(selected_classes,
                                    wnid_labels,
                                    original_training_path,
                                    new_training_path):
    class_list = []
    for i in range(0, 1000):
        folder = wnid_labels[i]
        if folder in selected_classes:
            continue
        class_list.append(folder)

    train_dst = os.path.join(new_training_path, 'gclass')

    cnt = 0
    chosen_elems = []
    # number of images from garbage classes = number of images from selected classes
    while (cnt <= 50):
        cls_num = random.randint(0, len(class_list) - 1)
        elem = class_list[cls_num]
        train_src = os.path.join(original_training_path, elem.strip('\n'))
        train_images = os.listdir(train_src)
        img_num = random.randint(0, len(train_images) - 1)
        img = train_images[img_num]
        src_img = os.path.join(train_src, img)
        dst_img = os.path.join(train_dst, img)
        if src_img in chosen_elems:
            continue
        else:
            cnt += 1
            dst_img = os.path.join(train_dst, str(cnt) + '.JPEG')
            if os.path.isfile(dst_img):
                os.remove(dst_img)
            os.symlink(src_img, dst_img)
            chosen_elems.append(src_img)


def create_garbage_class_folder(selected_classes,
                                wnid_labels,
                                original_training_path,
                                new_training_path,
                                original_val_path,
                                new_val_path,
                                garbage_multiplier=1):
    class_list = []
    for i in range(0, 1000):
        folder = wnid_labels[i]
        if folder in selected_classes:
            continue
        class_list.append(folder)
    if new_training_path is not '':
        train_dst = os.path.join(new_training_path, 'gclass')
        if os.path.exists(train_dst):
            shutil.rmtree(train_dst)
        os.makedirs(train_dst)
        change_garbage_class_folder(selected_classes,
                                    wnid_labels,
                                    original_training_path,
                                    new_training_path,
                                    garbage_multiplier=garbage_multiplier)
    if new_val_path is not '':
        val_dst = os.path.join(new_val_path, 'gclass')
        if os.path.exists(val_dst):
            shutil.rmtree(val_dst)
        os.makedirs(val_dst)
        cnt = 0
        total_valid = 0
        img_cnt = 0
        for elem in class_list:
            cnt += 1
            val_src = os.path.join(original_val_path, elem.strip('\n'))
            val_images = os.listdir(val_src)

            for img in val_images:
                src_img = os.path.join(val_src, img)
                dst_img = os.path.join(val_dst, img)
                os.symlink(src_img, dst_img)


def create_garbage_links(num_classes,
                         wnid_labels,
                         original_training_path,
                         new_training_path,
                         original_validation_path,
                         new_validation_path):

    if new_training_path is not '':
        if os.path.exists(new_training_path):
            shutil.rmtree(new_training_path)
        os.makedirs(new_training_path)

    if new_validation_path is not '':
        if os.path.exists(new_validation_path):
            shutil.rmtree(new_validation_path)
        os.makedirs(new_validation_path)

    # Not sure if it's zero based or 1 based in Keras => remove both 0 and 1000
    class_indices = random.sample(range(0, 999), int(num_classes) - 1)
    class_indices = sorted(class_indices)
    class_list = []

    if os.path.isfile('selected_dirs.txt'):
        class_indices = []  # just to be safe.
        class_list = []
        f = open('selected_dirs.txt', 'r')
        for line in f:
            class_list.append(line)
        f.close()
    else:
        for class_index in class_indices:
            folder = wnid_labels[class_index]
            class_list.append(folder)
        f = open('selected_dirs.txt', 'w')
        for elem in class_list:
            f.write(elem + '\n')
        f.close()

    for dir in class_list:
        if new_training_path is not '':
            src = os.path.join(original_training_path, dir.strip('\n'))
            dst = os.path.join(new_training_path, dir.strip('\n'))
            os.symlink(src, dst)
        if new_validation_path is not '':
            src = os.path.join(original_validation_path, dir.strip('\n'))
            dst = os.path.join(new_validation_path, dir.strip('\n'))
            os.symlink(src, dst)

    return class_list


def select_input_classes(num_classes,
                         wnid_labels,
                         original_training_path,
                         original_validation_path,
                         config_path):

    class_list = []
    c_path = os.path.join(config_path, "config.txt")
    if (os.path.isfile(c_path)):
        with open(c_path, "r") as ins:
            array = []
            for line in ins:
                if "Classes" in line:
                    class_list = line.replace("Classes:", "").replace(" ", "") \
                        .split(',')
                    if len(class_list) != num_classes:
                        print("\n\nlow number of classes in config file")
                        exit()
                    if not verify_classes(class_list, original_training_path, \
                                          original_validation_path):
                        print("\n\nlow number of figures in class")
                        exit()
                    return class_list

    while len(class_list) < num_classes:
        class_index = random.randrange(1000)
        folder = wnid_labels[class_index]
        if folder not in class_list and verify_classes([folder], \
                                                       original_training_path, original_validation_path):
            class_list.append(folder)

    if os.path.exists(c_path):
        a_w = 'a'  # append if already exists
    else:
        a_w = 'w'

    file = open(c_path, a_w)
    file.write('Classes: ' + (",".join(class_list)))

    return class_list


def verify_classes(class_list,
                   original_training_path,
                   original_validation_path):
    if not class_list:
        return True

    for dir in class_list:
        path = os.path.join(original_training_path, dir.strip('\n'))
        if len([name for name in os.listdir(path) if \
                os.path.isfile(os.path.join(path, name))]) < 1300:
            return False

    for dir in class_list:
        path = os.path.join(original_validation_path, dir.strip('\n'))
        if len([name for name in os.listdir(path) if \
                os.path.isfile(os.path.join(path, name))]) < 50:
            return False

    return True


def onehot(index, tot_classes):
    """ It creates a one-hot vector with a 1.0 in
        position represented by index
    """
    '''new_idx = 0
    if index >= tot_classes-1:
        new_idx = tot_classes-1
    else:
        new_idx = index'''
    onehot = np.zeros(tot_classes)
    onehot[index] = 1.0
    # print(onehot)
    return onehot


def preprocess_image(image_path, augment_img=False):
    """ It reads an image, it resize it to have the lowest dimesnion of 256px,
        it randomly choose a 227x227 crop inside the resized image and normalize
        the numpy array subtracting the ImageNet training set mean
        Args:
            images_path: path of the image
        Returns:
            cropped_im_array: the numpy array of the image normalized
            [width, height, channels]
    """
    IMAGENET_MEAN = [123.68, 116.779, 103.939]  # rgb format

    img = Image.open(image_path).convert('RGB')
    # resize of the image (setting lowest dimension to 256px)
    if img.size[0] < img.size[1]:
        h = int(float(256 * img.size[1]) / img.size[0])
        img = img.resize((256, h), Image.ANTIALIAS)
    else:
        w = int(float(256 * img.size[0]) / img.size[1])
        img = img.resize((w, 256), Image.ANTIALIAS)

    x = random.randint(0, img.size[0] - 227)
    y = random.randint(0, img.size[1] - 227)
    img_cropped = img.crop((x, y, x + 227, y + 227))
    cropped_im_array = np.array(img_cropped, dtype=np.float32)

    for i in range(3):
        cropped_im_array[:, :, i] -= IMAGENET_MEAN[i]

    if augment_img == True:
        '''
            Returns original cropped image and list of cropped images

                augment 'mode' parameter:
                    0 = flip only
                    1 = flip and rotate
                    2 = flip, rotate, and translate

                augmented_imgs[0] = flipped
                augmented_imgs[1] = rotated
                augmented_imgs[2] = translated
        '''
        augmented_imgs = augment(cropped_im_array, mode=0)
        return cropped_im_array, augmented_imgs

    '''
        Otherwise just returns original cropped image
    '''
    return cropped_im_array


def load_imagenet_meta(meta_path):
    """ It reads ImageNet metadata from ILSVRC 2012 dev tool file
        Args:
            meta_path: path to ImageNet metadata file
        Returns:
            wnids: list of ImageNet wnids labels (as strings)
            words: list of words (as strings) referring to wnids labels and
            describing the classes
    """
    metadata = loadmat(meta_path, struct_as_record=False)
    '''
    ['ILSVRC2012_ID', 'WNID', 'words', 'gloss', 'num_children', 'children', 
    'wordnet_height', 'num_train_images']
    '''
    synsets = np.squeeze(metadata['synsets'])
    ids = np.squeeze(np.array([s.ILSVRC2012_ID for s in synsets]))
    wnids = np.squeeze(np.array([s.WNID for s in synsets]))
    words = np.squeeze(np.array([s.words for s in synsets]))
    return wnids, words


def imagenet_size(im_source):
    """ It calculates the number of examples in ImageNet training-set
        Args:
            im_source: path to ILSVRC 2012 training set folder
        Returns:
            n: the number of training examples
    """
    n = 0
    for d in os.listdir(im_source):
        for f in os.listdir(os.path.join(im_source, d)):
            n += 1
    return n

def copyModel2Model(model_source, model_target, certain_layer=""):
    for l_tg, l_sr in zip(model_target.layers, model_source.layers):
        wk0 = l_sr.get_weights()
        l_tg.set_weights(wk0)
        if l_tg.name == certain_layer:
            break
    print("Model source was copied into Model target")


'''
    Used for RMSprop optimizer for learning rate decay defined in
    Inception V4 Paper  https://arxiv.org/abs/1602.07261
'''
class ExpDecayScheduler(Callback):
    def __init__(self, initial_lr, n_epoch, decay):
        '''
        :param initial_lr: Initial Learning Rate
        :param n_epoch: Every epoch to decay learning rate
        :param decay: Decay factor
        '''
        super(ExpDecayScheduler, self).__init__()
        self.initial_lr = initial_lr
        self.n_epoch = n_epoch
        self.decay = decay

    def on_epoch_begin(self, epoch, logs={}):
        old_lr = K.get_value(self.model.optimizer.lr)
        if epoch % self.n_epoch == 0 and epoch > 0:
            new_lr = self.initial_lr * np.exp(-self.decay * epoch)
            print("Decaying Learning Rate... New LR = " + str(new_lr))
            K.set_value(self.model.optimizer.lr, new_lr)
        else:
            K.set_value(self.model.optimizer.lr, old_lr)


class TrainByBranch(Callback):
    '''

        **Deprecated: Shown to be ineffective, remove

        Train each branch on Octopus individually by setting
        (trainable=False) for all branches except desired one.
        When epoch % epoch_limit == 0 , change trainable branch.
    '''

    def __init__(self, num_classes, epoch_limit):
        '''
        :param num_classes: Number of classes
        :param epoch_limit: Epoch Limit to train each branch
        '''
        super(TrainByBranch, self).__init__()
        self.num_classes = num_classes
        self.epoch_limit = epoch_limit
        self.base_name_arr = ['inception_4c/1x1', \
                              'inception_4c_3x3_reduce', \
                              'inception_4c/5x5_reduce', \
                              'inception_4c/3x3', \
                              'inception_4c/5x5', \
                              'inception_4c/pool_proj', \
                              'inception_4d_1x1', \
                              'inception_4d_3x3_reduce', \
                              'inception_4d_5x5_reduce', \
                              'inception_4d_3x3', \
                              'inception_4d_5x5', \
                              'inception_4d_pool_proj', \
                              'inception_4e_1x1', \
                              'inception_4e_3x3_reduce', \
                              'inception_4e_5x5_reduce', \
                              'inception_4e_3x3', \
                              'inception_4e_5x5', \
                              'inception_4e_pool_proj', \
                              'inception_5a_1x1', \
                              'inception_5a_3x3_reduce', \
                              'inception_5a_5x5_reduce', \
                              'inception_5a_3x3', \
                              'inception_5a_5x5', \
                              'inception_5a_pool_proj', \
                              'inception_5b_1x1', \
                              'inception_5b_3x3_reduce', \
                              'inception_5b_5x5_reduce', \
                              'inception_5b_3x3', \
                              'inception_5b_5x5', \
                              'inception_5b_pool_proj', \
                              '_loss3/classifier']
        self.branch_num = 0

    def on_epoch_begin(self, epoch, logs={}):
        if epoch % self.epoch_limit == 0:
            print("\n\nTRAINING BRANCH " + str(self.branch_num) + "\n\n")
            for n in self.base_name_arr:
                # Set current branch to true
                K.set_value(self.model.get_layer(str(self.branch_num) + n).trainable, True)
                # Set previous branch to false (if not first branch)
                if self.branch_num > 0:
                    K.set_value(self.model.get_layer(str(self.branch_num - 1) + n).trainable, False)

            self.branch_num += 1
            # Iterate through layers to double check 'trainable'
            # Comment out when debugged
            for layer in model.layers:
                print(layer, layer.trainable)

class OneCycle(Callback):
    def __init__(self, min_lr, max_lr, min_mom, max_mom, step_size, div):
        '''
        As defined by Smith in arXiv:1803.09820

        :param min_lr: Minimum Learning Rate
        :param max_lr: Maximum Learning Rate
        :param min_mom: Minimum Momentum
        :param max_mom: Maximum Momentum
        :param step_size: number of iterations per half-cycle
        :param div:
        '''
        super(OneCycle, self).__init__()
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.min_mom = min_mom
        self.max_mom = max_mom
        self.step_size = step_size
        self.div = div

    def on_batch_begin(self, batch, logs={}):

        iteration = float(K.get_value(self.model.optimizer.iterations))
        cycle = math.floor(1.0 + iteration / (2.0 * self.step_size))
        x = math.fabs(iteration / self.step_size - 2.0 * cycle + 1.0)
        print("\nIteration: " + str(iteration))
        print("Cycle: " + str(cycle))
        print("X: " + str(x))
        if iteration > 2 * self.step_size:
            # Set learning rate depending on iteration
            new_lr = self.min_lr - (self.min_lr - (self.min_lr / self.div)) * \
                     math.fabs(1.0 - x)
            # allow lr to decay further to several order of mag lower than min
            if iteration > (2 + 1) * self.step_size:
                new_lr = (self.min_lr / self.div)
            K.set_value(self.model.optimizer.lr, new_lr)
            # Set momentum to max after cycle
            new_mom = self.max_mom
            K.set_value(self.model.optimizer.momentum, new_mom)
            print("Setting learning rate: " + str(new_lr))
            print("Setting momentum: " + str(new_mom))
        else:
            print("Min_lr: " + str(self.min_lr))
            print("Max_lr: " + str(self.max_lr))
            print("Max_lr - Min_lr: " + str((self.max_lr - self.min_lr)))
            print("abs(1.0-x) : " + str(math.fabs(1.0 - x)))
            # Set learning rate depending on iteration
            new_lr = self.min_lr + (self.max_lr - self.min_lr) * math.fabs(1.0 - x)
            K.set_value(self.model.optimizer.lr, new_lr)
            # Set momentum depending on iteration
            new_mom = self.max_mom - (self.max_mom - self.min_mom) * \
                      math.fabs(1.0 - x)
            K.set_value(self.model.optimizer.momentum, new_mom)
            print("Setting learning rate: " + str(new_lr))
            print("Setting momentum: " + str(new_mom))


class SaveWeightsNumpy(Callback):
    def __init__(self, num_classes,
                 orig_model,
                 file_path,
                 period,
                 selected_classes,
                 wnid_labels,
                 orig_train_img_path,
                 new_training_path,
                 orig_val_img_path,
                 new_val_path,
                 weight_filename='weights.npy',
                 best_l_g_filename='max_l_g_weights.npy',
                 best_loc_filename='loc_weights.npy',
                 is_pruning=False,
                 finetuning=False,
                 multi_outputs=True):

        super(SaveWeightsNumpy, self).__init__()
        self.num_classes = num_classes
        self.file_path = file_path
        self.period = period
        self.selected_classes = selected_classes
        self.wnid_labels = wnid_labels
        self.orig_train_img_path = orig_train_img_path
        self.new_training_path = new_training_path
        self.orig_val_img_path = orig_val_img_path
        self.new_val_path = new_val_path
        self.weight_filename = weight_filename
        self.best_l_g_filename = best_l_g_filename
        self.best_loc_filename=best_loc_filename
        self.orig_model = orig_model
        self.finetuning = finetuning
        self.is_pruning = is_pruning
        self.multi_outputs = multi_outputs
        self.loc_acc = 0.0
        self.acc_sum = 0.0
        
        if self.is_pruning:
            import prune_utils as pu
            from tensorflow_model_optimization.sparsity import keras as sparsity
            self.pu = pu
            self.sparsity = sparsity
    
    def on_epoch_end(self, epoch, logs={}):
        print("Epoch: " + str(epoch))
        if ((epoch+1) % self.period) == 0 and epoch > 1:
            print("Computing local validation accuracy...")
            # print("\nOverall Results: ")
            # print(self.orig_model.evaluate_generator(self.val_data_test, steps= \
            # 	                       int(self.num_classes)))
            selected_dirs = []
            selected_dirs.append('')
            f = open('selected_dirs.txt', 'r')
            for line in f:
                line = line.strip('\n')
                selected_dirs.append(line)
            f.close()
            selected_dirs = sorted(selected_dirs)
            all_dirs = os.listdir(self.new_val_path)
            total_imgs = 0
            correct_imgs = 0
            for folder in all_dirs:
                correct_index = 0
                if folder in selected_dirs:
                    correct_index = selected_dirs.index(folder)
                else:
                    continue
                p = os.path.join(self.new_val_path, folder)
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
                    pred = self.model.predict(img)
                    # print (pred[1])
                    total_imgs += 1
                    # print(pred[1].shape)
                    if self.multi_outputs:
                        if np.argmax(pred[0]) == correct_index:
                            correct_imgs += 1
                    else:
                        if np.argmax(pred) == correct_index:
                            correct_imgs += 1
            local_acc = correct_imgs * 1.0 / total_imgs
    
            #if epoch % self.period == 0 and epoch >= self.period:
            print("Saving weights to: " + str(self.file_path))
            weights = self.model.get_weights()
            if '.npy' in self.weight_filename:
                np.save(self.file_path + self.weight_filename, weights, fix_imports=True)
            elif '.h5' in self.weight_filename:
                #self.model.save(self.file_path + self.weight_filename)
                '''cp_model = tf.keras.models.clone_model(self.model)
                stripped_model = sparsity.strip_pruning(cp_model)'''
                stripped_model = self.sparsity.strip_pruning(self.model)
                tf.keras.models.save_model(stripped_model, self.file_path + self.weight_filename, include_optimizer=True)
            else:
                print('Unable to save weights. Invalid file format')
            # Check if sum of both global and local accuracy is the highest.
        
            global_local_sum = local_acc + logs.get('val_categorical_accuracy')

            print("\nglobal_local_sum: " + str(global_local_sum))
            if global_local_sum > self.acc_sum:
                self.acc_sum = global_local_sum
                print("Has highest acc_sum. Saving to max_l_g_weights.npy...\n")
                if '.npy' in self.best_l_g_filename:
                    np.save(self.file_path + self.best_l_g_filename, weights, fix_imports=True)
                elif '.h5' in self.best_l_g_filename:
                    #self.model.save(self.file_path + self.best_l_g_filename)
                    '''cp_model = tf.keras.models.clone_model(self.model)
                    stripped_model = sparsity.strip_pruning(cp_model)'''
                    stripped_model = self.sparsity.strip_pruning(self.model)
                    tf.keras.models.save_model(stripped_model, self.file_path + self.best_l_g_filename, include_optimizer=True)
                else:
                    print('Unable to save weights. Invalid file format')
            if local_acc >= self.loc_acc:
                self.loc_acc = local_acc
                print("Has high local acc. Saving weights now")
                if '.npy' in self.best_loc_filename:
                    np.save(self.file_path + self.best_loc_filename, weights, fix_imports=True)
                elif '.h5' in self.best_loc_filename:
                    #self.model.save(self.file_path + self.best_loc_filename)
                    '''cp_model = tf.keras.models.clone_model(self.model)
                    stripped_model = sparsity.strip_pruning(cp_model)'''
                    stripped_model = self.sparsity.strip_pruning(self.model)
                    tf.keras.models.save_model(stripped_model, self.file_path + self.best_loc_filename, include_optimizer=True)

            if self.is_pruning:
                sparsity_val = self.pu.calculate_sparsity(self.model)
                print("Current Sparsity: %f" % sparsity_val)
                with open(self.file_path + 'sparsity_pruning_logs.txt', 'a+') as f:
                    f.write('Epoch: %d\n' % epoch)
                    f.write('Sparsity: %f\n' % sparsity_val)
