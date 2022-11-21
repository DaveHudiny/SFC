#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: David Hudak
# File: main.py
# Login: xhudak03
# Course: SFC
# School: BUT FIT
# Short description: This file contains implementation of neural network

import numpy as np
from functions import *
import pickle
import matplotlib.pyplot as plt
from layer import Layer


class NNET:
    """
    Universal neural network class

    Parameters
    ----------
    input_size : int
        Size of neural network input
    layer_types : list of function references
        Determines activation functions of each layer (for example: [tanh, ReLU, soft_max])
    layer_sizes : list of ints
        Determines sizes of each layer (have to be of same length as layer_types)
    layer_ders : list of function references
        Determines derivations of activation functions (have to be same lenghts as layer_types)
    default_weights : np.array
        Default weights for each layer of neural network
    default_biases : np.array
        Default biases for each layer of neural network
    object_func : function reference
        Determines which loss function to use
    object_func_der : function reference
        Determines which loss function derivation for learning to use
    debuff : float
        Determines debuff for weight sizes -- sometimes the output of neural network is too because of
        high values of network weights. This debuff makes: weights *= debuff
    name : str
        Determines name of neural network
    """

    def __init__(self, input_size=0, layer_types=[], layer_sizes=[], layer_ders=[], default_weights=None,
                 default_biases=None, object_func=cross_entropy, object_func_der=cross_entropy_der, debuff=1.0,
                 name="Custom"):
        # self.number_of_layers = number_of_layers
        # self.layer_sizes = layer_sizes
        # self.layer_types = layer_types

        self.name = name
        self.input_size = input_size
        self.layers = []
        self.output_size = layer_sizes[-1]
        for i in range(len(layer_sizes)):
            if default_weights != None:
                weights = default_weights[i]
                bias = default_biases[i]
            elif i == 0:
                weights = np.random.rand(input_size, layer_sizes[i]) - 0.5
                weights *= debuff
                bias = np.random.rand(1, layer_sizes[i]) - 0.5
            elif i > 0:
                weights = 2 * \
                    np.random.rand(layer_sizes[i - 1], layer_sizes[i]) - 1
                weights *= debuff
                bias = 2*np.random.rand(1, layer_sizes[i]) - 1
            self.layers.append(Layer("FC", weights, bias, None, None))
            self.layers.append(
                Layer("Act", weights, bias, layer_types[i], layer_ders[i]))
        self.object_func = object_func
        self.object_func_der = object_func_der

    def debuffing(self, debuff):
        for layer in self.layers:
            if layer.type == "FC":
                layer.weights *= debuff

    def load_nnet(file):
        with open(file, "rb") as inp:
            try:
                nnet = pickle.load(inp)
            except:
                return None
            if type(nnet) != NNET:
                print(type(nnet))
                return None
            return nnet

    def save_nnet(self, file):
        with open(file, 'wb') as save:
            pickle.dump(self, save, pickle.HIGHEST_PROTOCOL)

    def __str__(self):
        return self.name

    def forward_propagation(self, input):
        for layer in self.layers:
            input = layer.forward_propagation(input)
        return input

    def predict(self, input):
        """
        Function predicts class of input.

        Parameters
        ----------
        input : np.array [[0, 1]]
            input of neural network
        """
        output = self.forward_propagation(input)
        return np.argmax(soft_max(output)), soft_max(output)

    def backward_propagation(self, input, learning_rate):
        for layer in reversed(self.layers[:-1]):
            input = layer.backward_propagation(input, learning_rate)
        return input

    def learn(self, inputs, labels, iterations, learning_rate, learn_image="img.png", error_print=20):
        """
        Function learns neural network on input labels

        Parameters
        ----------
        inputs : 
        """
        errors = []
        for i in range(iterations):
            error_val = 0
            for input, label in zip(inputs, labels):
                output = self.forward_propagation(input)
                # error = self.object_func_der(label, output)
                logits = soft_max(output)
                error_val += self.object_func(logits, label)
                error = self.object_func_der(output, label)
                self.backward_propagation(error, learning_rate)
            errors.append(error_val)
            if i % error_print == 0:
                print(f"Epocha: {i}, error: {error_val}")

        plt.plot(errors)
        plt.xlabel("Počet epoch")
        plt.ylabel("Velikost chyby")
        plt.title("Celková chyba dle epoch")
        plt.savefig(learn_image)


if __name__ == "__main__":
    network = NNET(input_size=2, layer_types=[tanh, ReLU], layer_sizes=[10, 2],
                   layer_ders=[tanh_der, ReLU_der], object_func=mse, object_func_der=mse_der)
    x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
    y_train = np.array([[[1, 0]], [[0, 1]], [[0, 1]], [[1, 0]]])
    # network = NNET.load_nnet("file.pkl")
    network.learn(x_train, y_train, 1000, 0.1)
    count_success(network, x_train, y_train)
    network.save_nnet("file.pkl")
    # print(cross_entropy(network.forward_propagation(np.array([1, 0])), np.array([1, 0, 0, 0, 0])))
    pass
