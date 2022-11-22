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

def load_custom_task(ftrain_x, ftrain_y, ftest_x, ftest_y):
    tr_x = open(ftrain_x, "r")
    tr_y = open(ftrain_y, "r")
    te_x = open(ftest_x, "r")
    te_y = open(ftest_y, "r")
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    linesx = tr_x.readlines()
    linesy = tr_y.readlines()
    if len(linesx) != len(linesy):
        assert()
    for trainx, trainy in zip(linesx, linesy):
        linex = np.array([trainx.split(" ")]).astype(np.float)
        liney = np.array([trainy.split(" ")]).astype(np.float)
        train_x.append(linex)
        train_y.append(liney)
    linesx = te_x.readlines()
    linesy = te_y.readlines()
    if len(linesx) != len(linesy):
        assert()
    for testx, testy in zip(linesx, linesy):
        linex = np.array([testx.split(" ")]).astype(np.float)
        liney = np.array([testy.split(" ")]).astype(np.float)
        test_x.append(linex)
        test_y.append(liney)
    tr_x.close()
    tr_y.close()
    te_x.close()
    te_y.close()
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    return train_x, train_y, test_x, test_y

def learn(task, train_x, train_y, learning_rate : float, image : str, network: NNET, 
          iterations, how_often_error, split = False, how_split = 1, metapochs = 1):
    success = False
    while success != True:
        try:
            print("Trénink začal.")
            if split == True:
                splitor = int(train_x.shape[0] / how_split)
                for i in range(metapochs):
                    print(f"Nadepocha: {i + 1}")
                    for j in range(how_split):
                        print(f"Část datasetu: {j + 1}")
                        network.learn(train_x[j*splitor:(j+1)*splitor], train_y[j*splitor:(j+1)*splitor], 
                                    iterations, learning_rate, image, how_often_error)
            else: 
                network.learn(train_x, train_y, iterations, learning_rate, image, how_often_error)
            success = True
        except KeyboardInterrupt:
            print("Program byl silou ukončen.")
            exit(0)
        except :
            network.debuffing(debuff=0.5)
            print("Bylo nutno snížit iniciální váhy")
    print("Trénování bylo dokončeno")
    input("Pro pokračování stiskněte klávesu enter...")
    return network

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

    return train_x, train_y, test_x, test_y

    network = NNET(input_size = 784, layer_types = [ReLU, soft_max], layer_sizes=[512, 10], 
                   layer_ders=[ReLU_der, softmax_der])
    learn("MNIST", train_x[0:1000], train_y[0:1000], 0.0019, "img.png", network, 10, 1, True, 4, 2)
    train_x = train_x.reshape(train_x.shape[0], 1, 28*28)
    train_x = train_x.astype('float32')
    test_x = test_x.reshape(test_x.shape[0], 1, 28*28)
    test_x = test_x.astype('float32')
    train_x /= 255
    test_x /= 255
    train_x -= 0.5
    test_x -= 0.5

    count_success(network, test_x, test_y)
    return 
    success = False
    debuff = 0.5
    while success == False: 
        try:
            for i in range(2):
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
    network = NNET(input_size = 10, layer_types = [tanh, ReLU, ReLU, soft_max], layer_sizes = [150, 100, 100, 10], layer_ders = [tanh_der, ReLU_der, ReLU_der, softmax_der])

    network.learn(train_x, train_y, 300, 0.001)
    count_success(network, test_x, test_y)
    network.save_nnet("majority10.pkl")
