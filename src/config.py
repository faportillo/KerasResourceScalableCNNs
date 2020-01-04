from __future__ import print_function
import sys
import os
import numpy as np

sys.path.append("..")

'''****************************************************************************************************************
    ToDo:
    Need to be able to utilize MC Knapsack and automatically generate list of OFMs
    based on parameter count; a function of a lambda value and number of classes
****************************************************************************************************************'''
lambda_val = 0.25

'''****************************************************************************************************************
    STEP 1
    Supported model types (Note, if model as '*_rs' suffix, then it is a Resource-Scalable model:
        googlenet / googlenet_rs
        mobilenet / mobilenet_rs
****************************************************************************************************************'''
model_type = 'mobilenet_rs_layer'
machine_name = 'Instance1' # Name of machine for training, in case training on multiple machines
# Creates prefixes for symbolic links when creating training data, in case using shared file system to pull data
symlnk_prfx = 'EVAL1_GARBAGE'
load_weights = False # Used if loading weights to resume training, in case it stops prematurely

'''****************************************************************************************************************
    STEP 2
    Booleans to determine what you want to do with the model
****************************************************************************************************************'''
model_train = True
model_eval = False
model_prune = False
model_finetune = False
model_quantize = False

'''****************************************************************************************************************
    STEP 3
    Imagenet Dataset paths.
    
    IMAGENET_PATH : Main directory that contains both Imagenet Training and Validation data
    TRAIN_PATH : Subdirectory within IMAGENET_PATH that contains TRAINING data separated in subdirectories by category
    VAL_2_PATH = : Subdirectory within IMAGENET_PATH that contains VALIDATION data separated in subdirectories
                   by category
    META_FILE : Path that contains meta.mat file, usually in the *dev_kit* directory
    CONFIG_PATH : Config path, **DEPRECATED
    VALID_TIME_MINUTE : **DEPRECATED
****************************************************************************************************************'''
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

'''****************************************************************************************************************
    STEP 4
    If training: model_path = directory that will store the trained model weights and related files
    If validating/finetuning/pruning: model_path = directory where weights are saved.
****************************************************************************************************************'''
model_path = './L5_25p_layer/'
tb_logpath= model_path + "/logs" # Directory for TensorBoard Log Files

'''****************************************************************************************************************
    STEP 5
    Determine type of data. Options include: 
    'generator' -> Used if data is stored in separate folders by category (Utilizes Keras' ImageDataGenerator)
    'tfrecord' -> Used to create a TFRecord file **Not fully supported
****************************************************************************************************************'''
format='generator'

'''****************************************************************************************************************
    STEP 6
    Model training hyperparameters (For training, pruning, finetuning)
****************************************************************************************************************'''
num_classes = 5  # number of classes ( _rs models support 5,10,15,20 classes. Vanilla models support 1000 classes )
num_epochs = 1000  # Number of training epochs
optimizer = 'adam'  # Supported types: Adam, RMSProp, SGD (More to be added later)
learning_rate = 0.001 # Optimizer learning rate
batch_size = 64  # Training Batch size
val_batch_size = 64 # Validation_Batch Size
validation_period = 10 # Period to do validation step (i.e. if =10, evaluate on validation data every 10 epochs)
augment_data = True  # Augment data or not
garbage_multiplier = 6  # when =n loads 1300*n garbage data per epoch, the higher = more misc class data during training
workers = 4  # Number of workers/threads for multiprocessing data (Based on number of CPU cores)
max_queue_size = 31  # Data loading queue of batches before passing to GPU
use_aux = False  # **DEPRECATED

'''****************************************************************************************************************
    STEP 7 (If pruning, skip if not)
    Pruning-specific parameters
****************************************************************************************************************'''
prune_weight_file = 'max_l_g_weights.npy' # Trained model weights to prune with
do_orig_eval = False # If you want to do eval on original models before pruning to get the Global and Local Accuracies
initial_sparsity = 0.94
final_sparsity = 0.95
begin_step = 0
end_step = np.ceil(1.0 * ((num_classes - 1) * 1300) + (1300 * garbage_multiplier) / batch_size).astype(np.int32) * (num_epochs-2)
prune_frequency = 200
stopping_patience = 4
schedule = 'polynomial'  # Supports 'polynomial' and 'constant' pruning

'''****************************************************************************************************************
    STEP 8 (If doing eval, skip if not)
    Evaluation-specific parameters
    Used to changed names of files when writing local and global accuracy to files
****************************************************************************************************************'''
eval_weight_file = 'max_l_g_weights.npy'
is_pruned = True  # True if evaluating a pruned model
is_finetuned = False  # True if evalauating a finetuned model


'''****************************************************************************************************************
    Automatically set variables and safety checks. DO NOT TOUCH
****************************************************************************************************************'''
if model_type == 'googlenet' or model_type == 'mobilenet':
    image_size = 224
    num_classes = 1000  # HAS to be 1000 classes for vanilla model types
    do_orig_eval = False
elif model_type == 'googlenet_rs' or model_type == 'mobilenet_rs' or model_type == 'mobilenet_rs_layer':
    image_size = 227

