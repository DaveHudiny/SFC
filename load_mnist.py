#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: David Hudak
# File: main.py
# Login: xhudak03
# Course: SFC
# School: BUT FIT
# Short description: This file contains loader of mnist datasets. Use only if you want to download whole mnist dataset

from keras.datasets import mnist
from keras.utils import np_utils
import pickle 
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
x_train = x_train.astype('float32')
x_test = x_test.reshape(x_train.shape[0], 1, 28*28)
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
x_train -= 0.5
x_test -= 0.5
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

with open("train_x.pkl", 'wb') as save:
  pickle.dump(x_train, save, pickle.HIGHEST_PROTOCOL)
with open("train_y.pkl", 'wb') as save:
  pickle.dump(y_train, save, pickle.HIGHEST_PROTOCOL)
with open("test_x.pkl", 'wb') as save:
  pickle.dump(x_test, save, pickle.HIGHEST_PROTOCOL)
with open("test_y.pkl", 'wb') as save:
  pickle.dump(y_test, save, pickle.HIGHEST_PROTOCOL)