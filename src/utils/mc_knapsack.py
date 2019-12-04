from __future__ import print_function
import sys

sys.path.append("..")
import tensorflow as tf
import numpy as np
import train_utils as tu
import eval_utils as eu

import os
import math
import time

'''
    Implementation of the MC-Knapsack problem for finding the optimal scaling 
    factors given a desired parameter budget.
    
    - Each macro-layer is a class 
    - Parameter budget, Phi', is the knapsack size
    - Only one scaling factor can be selected per class
    - Size and reward associated with an item are equal to the scaled number
      of parameters in the macro-layer.
    - Follows additional bottleneck avoidance constraint where # of channels in current
      macro-layer need to be higher than previous layers.
'''
class MCKP():
    def __init__(self):
        super(MCKP, self).__init__()

    def read_model_structure(self, model_file):
        '''

        :param model_file: Filepath to model json description
        :return:
        '''



class MacroLayer():
    def __init__(self, kernel_list, ifm_list, ofm_list):
        super(MacroLayer, self).__init__()
        self.kernel_list = np.array(kernel_list)
        self.ifm_list = np.array(ifm_list)
        self.ofm_list = np.array(ofm_list)

    def calc_parameters(self, scaling_factor=1):
        scaled_ofms = np.rint(self.ofm_list / scaling_factor)
        return np.sum(np.multiply(np.multiply(np.multiply(self.kernel_list,
                                                   self.kernel_list),
                                       self.ifm_list), scaled_ofms))