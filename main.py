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

current_network = None
loss_img = "loss.png"


def print_loaded():
    print(f"Vítejte v programu pro práci s neuronovými sítěmi. \n\nAktuální stav globálních proměnných:")
    print(f"  Právě máte načtenou síť: {current_network}")
    print(f"  Grafy vývoje chybové funkce se aktuálně nachází v souboru: {loss_img}")
    print(f"  ")

def print_menu():
    print("Co si přejete dělat?")
    print("1) Vybrat úlohu")
    print("C) Vytvořit vlastní síť")
    print("L) Načíst síť")
    print("S) Uložit síť")
    print("Q) Ukončit program")
    print()
    pass

def load_network():
    file = input("Zadejte vstupní soubor: ")
    if len(file) == 0 or os.path.exists(file) != True:
        print("Nezadal jste korektní název.")
        time.sleep(2)
        return
    global current_network
    current_network = NNET.load_nnet(file)
    if current_network == None:
        print("Nepovedlo se načíst síť.")
        time.sleep(2)

def save_network():
    if current_network == None:
        print("Není co uložit")
        time.sleep(2)
    else:
        file = input("Zadejte jméno souboru: ")
        try:
            current_network.save_nnet(file=file)
        except:
            print("Ukládání se nezdařilo")
            time.sleep(2)
            return
        print(f"Síť uložena do {file}")
        time.sleep(2)

def create_network():
    input_size = input("Zadejte počet vstupů sítě: ")
    try:
        input_size = int(input_size)
    except:
        print("Nejedná se o legální hodnotu")
        time.sleep(2)
        return
    number_of_layers = input("Zadejte počet vrstev sítě: ")
    try:
        number_of_layers = int(number_of_layers)
    except:
        print("Nejedná se o legální hodnotu")
        time.sleep(2)
        return
    layer_sizes = []
    for i in range(number_of_layers):
        layer_size = input(f"Počet neuronů v uzlu {i + 1}: ")
        try:
            layer_size = int(layer_size)
        except:
            print("Nejedná se o legální hodnotu")
            time.sleep(2)
            return
        layer_sizes.append(layer_size)

        

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
        if os.name == "nt":
            os.system("CLS")
        else:
            os.system("clear")
        print_loaded()
        print_menu()
        text = input("Vaše volba: ")
        out = selection(text)
        if out == 1:
            end = True
