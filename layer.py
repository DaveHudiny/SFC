#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: David Hudak
# File: main.py
# Login: xhudak03
# Course: SFC
# School: BUT FIT
# Short description: This file contains implementation of layers of neural network


import numpy as np
from functions import *

class Layer:
    def __init__(self, type, weights, biases, activation, activation_der):
        self.type = type
        if type == "FC":
            self.weights = weights
            self.biases = biases
            self.momentum = 0
            self.momentum_bias = 0
            self.alpha = 0.6
        elif type == "Act":
            self.activation = activation
            self.activation_der = activation_der
    
    def __str__(self):
        if self.type == "FC":
            print("Váhy", self.weights)
            print("Biasy", self.biases)
            return ""
        elif self.type == "Act":
            print(self.activation(0))
            return ""
        else:
            return "Unknown"
    
    def forward_propagation(self, layer_input):
        if self.type == "FC":
            self.input_mem = layer_input
            self.output_mem = np.dot(layer_input, self.weights) + self.biases
            return self.output_mem
        elif self.type == "Act":
            self.input_mem = layer_input
            self.output_mem = self.activation(layer_input)
            return self.output_mem
        else:
            print("Dont know this layer")
    
    def backward_propagation(self, output_error, learning_rate):
        if self.type == "FC":
            input_error = np.dot(output_error, self.weights.T)
            weights_error = np.dot(self.input_mem.T, output_error)
            # change_weight = alpha * self.momentum - learning_rate * weights_error
            self.weights -= learning_rate * weights_error
            # self.momentum_weights = change_weight
            # change_bias = alpha * self.momentum_bias - 
            self.biases -= learning_rate * output_error
            # self.momentum_bias = change_bias
            return input_error
        elif self.type == "Act":
            return self.activation_der(self.input_mem) * output_error
