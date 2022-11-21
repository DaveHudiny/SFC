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
import custom_net as cn

current_network = None
loss_img = "loss.png"
input_size = -1
task = None

train_x, train_y , test_x, test_y = None, None, None, None


def print_loaded():
    print(f"Vítejte v programu pro práci s neuronovými sítěmi. \n\nAktuální stav globálních proměnných:")
    print(f"  Právě máte načtenou síť: {current_network}")
    print(f"  Grafy vývoje chybové funkce se aktuálně nachází v souboru: {loss_img}")
    print(f"  Počet aktuálních vstupů sítě/úlohy {input_size} (pokud -1, vstup nebyl určen)")
    print(f"  Aktuální úloha {task}")

def print_menu():
    print("Co si přejete dělat?")
    print("T) Vybrat úlohu")
    print("C) Vytvořit vlastní síť")
    print("L) Načíst síť")
    print("S) Uložit síť")
    print("Q) Ukončit program")
    print()
    pass

def choose_task():
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

def selection(x):
    if len(x) == 0 or x[0] in ["q", "Q"]:
        return 1
    elif x[0] in ["l", "L"]:
        load_network()
    elif x[0] in ["s", "S"]:
        save_network()
    elif x[0] in["c", "C"]:
        global current_network
        current_network = cn.create_network()
        if current_network != None:
            input_size = current_network.input_size
    elif x[0] in ["t", "T"]:
        choose_task()
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
