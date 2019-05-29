"""Custom Keras layers for GoogLeNet"""
import tensorflow as tf
from tensorflow.python.keras.layers.core import Layer
from tensorflow.python.keras.engine import InputSpec
from tensorflow.python.keras import backend as K
try:
    from keras import initializations
except ImportError:
    from keras import initializers as initializations

def LRN(x):
    return tf.nn.local_response_normalization(x)


class PoolHelper(Layer):
    """
    Reconcile Keras and Caffe weights
    """

    def __init__(self, **kwargs):
        super(PoolHelper, self).__init__(**kwargs)
    
    def call(self, x, mask=None):
        return x[:,:,1:,1:]
    
    def get_config(self):
        config = {}
        base_config = super(PoolHelper, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))