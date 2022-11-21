#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: David Hudak
# File: main.py
# Login: xhudak03
# Course: SFC
# School: BUT FIT
# Short description: This file contains useful tools for SFC project implementation

import numpy as np
from nnet import *
import pickle
from functions import count_success

np.seterr(all='raise')

def generate_majority(samples_number = 200, input_size = 3, output_size = 3):
    train_x = []
    train_y = []
    if input_size <= 0 or output_size <= 0:
        return train_x, train_x
    for i in range(samples_number):
        rand_arr = np.random.randint(0, output_size, input_size)
        dictionary = {}
        for i in rand_arr:
            if i not in dictionary.keys():
                dictionary[i] = 1
            else:
                dictionary[i] += 1
        new = np.array([rand_arr])
        train_x.append(new)
        maximum = max(dictionary, key=dictionary.get)
        new_y = np.zeros((output_size,))
        new_y[maximum] = 1.0
        new_y = np.array([new_y])
        train_y.append(new_y)
    return np.array(train_x), np.array(train_y)

def load_mnist():
    with open("./train/train_mnist/train_x.pkl", "rb") as inp:
        train_x = pickle.load(inp)
    with open("./train/train_mnist/train_y.pkl", "rb") as inp:
        train_y = pickle.load(inp)
    with open("./test/test_mnist/test_x.pkl", "rb") as inp:
        test_x = pickle.load(inp)
    with open("./test/test_mnist/test_y.pkl", "rb") as inp:
        test_y = pickle.load(inp)
    train_x = train_x / 255.0
    train_x -= 0.5
    test_x = test_x / 255.0
    test_x -= 0.5

    network = NNET(input_size = 784, layer_types = [ReLU, soft_max], layer_sizes=[512, 10], 
                   layer_ders=[ReLU_der, softmax_der])
    success = False
    debuff = 0.5
    while success == False: 
        try:
            for i in range(10):
                print(f"Metapocha {i}")
                network.learn(train_x[0:1000], train_y[0:1000], 4, learning_rate=0.0019)
                network.learn(train_x[5000:6000], train_y[5000:6000], 4, learning_rate=0.0019)
                network.learn(train_x[3000:4000], train_y[3000:4000], 4, learning_rate=0.0019)
                network.learn(train_x[12000:13000], train_y[12000:13000], 4, learning_rate=0.0019)
            success = True
        except KeyboardInterrupt:
            print("Program byl silou ukončen.")
            exit(0)
        except:
            network = NNET(input_size = 784, layer_types = [ReLU, ReLU, ReLU, soft_max], layer_sizes=[256, 256, 256, 10], 
                           layer_ders=[ReLU_der, ReLU_der, ReLU_der, softmax_der], debuff=debuff)
            debuff *= 0.5
            print("Bylo nutno snížit iniciální váhy")
    count_success(network, test_x[0:1000], test_y[0:1000])
    network.save_nnet("mnist_network.pkl")

# load_mnist()
# exit()
if __name__ == "__main__":
    train_x, train_y = generate_majority(samples_number=5000, input_size=10, output_size=10)
    test_x, test_y = generate_majority(samples_number=1000, input_size=10, output_size=10)
    train_x = train_x / 10.0
    test_x = test_x / 10.0
    train_x -= 0.5
    test_x -= 0.5
    network = NNET(input_size = 10, number_of_layers = 4, layer_types = [tanh, ReLU, ReLU, soft_max], layer_sizes = [150, 100, 100, 10], layer_ders = [tanh_der, ReLU_der, ReLU_der, softmax_der])

    network.learn(train_x, train_y, 300, 0.001)
    count_success(network, test_x, test_y)
    network.save_nnet("majority10.pkl")
