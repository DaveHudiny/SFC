#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: David Hudak
# File: main.py
# Login: xhudak03
# Course: SFC
# School: BUT FIT
# Short description: This file contains UI implementation for project to course SFC

import os
import time

from nnet import NNET
import tools
from functions import *

current_network = None
loss_img = "loss.png"


def print_loaded():
    print(f"Vítejte v programu pro práci s neuronovými sítěmi. \n\nAktuální stav globálních proměnných:")
    print(f"  Právě máte načtenou síť: {current_network}")
    print(f"  Grafy vývoje chybové funkce se aktuálně nachází v souboru: {loss_img}")
    print(f"  ")

def print_menu():
    print("Co si přejete dělat?")
    print("T) Vybrat úlohu")
    print("C) Vytvořit vlastní síť")
    print("L) Načíst síť")
    print("S) Uložit síť")
    print("Q) Ukončit program")
    print()
    pass

def choose_tast():
    print("Vyberte jednu z následujících úloh:")
    print("  m) MNIST")



def load_network():
    file = input("Zadejte vstupní soubor: ")
    if len(file) == 0 or os.path.exists(file) != True:
        print("Nezadal jste korektní název.")
        input("\nStiskněte enter pro pokračování...")
        return
    global current_network
    current_network = NNET.load_nnet(file)
    if current_network == None:
        print("Nepovedlo se načíst síť.")
        input("\nStiskněte enter pro pokračování...")

def save_network():
    if current_network == None:
        print("Není co uložit")
        input("\nStiskněte enter pro pokračování...")
    else:
        file = input("Zadejte jméno souboru: ")
        try:
            current_network.save_nnet(file=file)
        except:
            print("Ukládání se nezdařilo")
            input("\nStiskněte enter pro pokračování...")
            return
        print(f"Síť uložena do {file}")
        input("\nStiskněte enter pro pokračování...")

def clear_screen():
    if os.name == "nt":
        os.system("CLS")
    else:
        os.system("clear")

def activation_list_hidden():
    print("Možnosti aktivačních funkcí skrytých vrstev:")
    print("  r) ReLU (výchozí)")
    print("  s) Sigmoid")
    print("  t) Tanh")

def activation_list_output():
    print("Možnosti aktivačních funkcí výstupních vrstev:")
    print("  m) Softmax (výchozí)")
    print("  s) Sigmoid")
    print("  l) Lineární")

def loss_function_list():
    print("Možnosti chybových funkcí:")
    print("  c) Křížová entropie (výchozí)")
    print("  m) Střední kvadratická odchylka")

def choose_function(x):
    if len(x) <= 0:
        return ReLU, ReLU_der
    elif x[0] in ["t", "T"]:
        return tanh, tanh_der
    elif x[0] == ["s", "S"]:
        return sigmoid, sigmoid_der
    else:
        return ReLU, ReLU_der

def choose_output_function(x):
    if len(x) <= 0:
        return soft_max, softmax_der
    elif x[0] in ["l", "L"]:
        return linear, linear_der
    elif x[0] in ["s", "S"]:
        return sigmoid, sigmoid_der
    else:
        return soft_max, softmax_der

def choose_loss_function(x):
    if len(x) <= 0:
        return cross_entropy, cross_entropy_der
    elif x[0] in ["m", "M"]:
        return mse, mse_der
    else:
        return cross_entropy, cross_entropy_der

def create_network():
    input_size = input("Zadejte počet vstupů sítě: ")
    try:
        input_size = int(input_size)
    except:
        print("Nejedná se o legální hodnotu")
        input("\nStiskněte enter pro pokračování...")
        return
    number_of_layers = input("Zadejte počet skrytých vrstev sítě: ")
    try:
        number_of_layers = int(number_of_layers)
    except:
        print("Nejedná se o legální hodnotu")
        input("\nStiskněte enter pro pokračování...")
        return
    layer_sizes = []
    layer_types = []
    layer_types_der = []
    activation_list_hidden()
    for i in range(number_of_layers):
        layer_size = input(f"Počet neuronů ve vrstvě {i + 1}: ")
        try:
            layer_size = int(layer_size)
        except:
            print("Nejedná se o legální hodnotu")
            input("\nStiskněte enter pro pokračování...")
            return
        layer_sizes.append(layer_size)
        layer_type = input(f"Aktivační funkce skryté vrstvy {i + 1}: ")
        func, func_der = choose_function(layer_type)
        layer_types.append(func)
        layer_types_der.append(func_der)
    number_of_outputs = input("Počet výstupů sítě: ")
    try:
        number_of_outputs = int(number_of_outputs)
    except:
        print("Nejedná se o legální hodnotu")
        input("\nStiskněte enter pro pokračování...")
        return
    layer_sizes.append(number_of_outputs)
    activation_list_output()
    output_function = input("Zvolte výstupní aktivační funkci: ")
    output, output_der = choose_output_function(output_function)
    layer_types.append(output)
    layer_types_der.append(output_der)
    loss_function_list()
    loss = input("Zvolte chybovou (ztrátovou) funkci: ")
    loss_fun, loss_fun_der = choose_loss_function(loss)
    name = input("Zvolte jméno své sítě: ")
    global current_network
    current_network = NNET(input_size, layer_types, layer_sizes, layer_types_der, None, None, loss_fun, loss_fun_der, 1, name)
    x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
    y_train = np.array([[[1, 0]], [[0, 1]], [[0, 1]], [[1, 0]]])
    current_network.learn(x_train, y_train, 1000, 0.1)
    tools.count_success(current_network, x_train, y_train)
    print(f"Úspěšně jste vytvořil neuronovou síť {name}")
    print()
    input("Stiskněte enter pro pokračování...")

def selection(x):
    if len(x) == 0 or x[0] in ["q", "Q"]:
        return 1
    elif x[0] in ["l", "L"]:
        load_network()
    elif x[0] in ["s", "S"]:
        save_network()
    elif x[0] in["c", "C"]:
        create_network()
    return 0
        

if __name__ == "__main__":
    end = False
    while end == False:
        clear_screen()
        print_loaded()
        print_menu()
        text = input("Vaše volba: ")
        out = selection(text)
        if out == 1:
            end = True
