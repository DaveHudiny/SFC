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
output_size = -1
task = None

train_x, train_y , test_x, test_y = None, None, None, None


def print_loaded():
    print(f"Vítejte v programu pro práci s neuronovými sítěmi. \n\nAktuální stav globálních proměnných:")
    print(f"  Právě máte načtenou síť: {current_network}")
    print(f"  Grafy vývoje chybové funkce se aktuálně nachází v souboru: {loss_img}")
    print(f"  Počet aktuálních vstupů sítě/úlohy {input_size} (pokud -1, vstup nebyl určen)")
    print(f"  Počet aktuálních výstupů sítě/úlohy {input_size} (pokud -1, výstup nebyl určen)")
    print(f"  Aktuální úloha {task}")

def print_menu():
    print("Co si přejete dělat?")
    print("T) Vybrat úlohu")
    print("C) Vytvořit vlastní síť")
    print("L) Načíst síť")
    print("S) Uložit síť")
    print("E) Uč síť")
    print("Q) Ukončit program")
    print()
    pass

def print_choose_task(with_help = True):
    print("Vyberte jednu z následujících úloh:")
    print("  m) MNIST")
    print("  o) Výpočet modu (nejčastější hodnoty)")
    print("  x) XOR problém")
    print("  c) Vlastní problém")
    if with_help:
        print("  h) Jak na formát vlastního formátu?")
    print()

def help_choose():
    print("\n V případě volby vlastního formátu musíte mít alespoň jeden soubor se vstupy (pokud trénovací data == testovací data) a jeden s výstupy.")
    print("Soubor se vstupy musí obsahovat posloupnost desetinných čísel od -1.0 do 1.0 oddělených mezerami, kde řádek znamená jeden vstupní vektor.")
    print("Soubor s výstupy musí obsahovat posloupnost desetinných čísel od 0.0 do 1.0 (příslušnost) oddělených mezerami, kde řádek symbolizuje vektor výstupu pro vstup.")
    print("První řádek ve vstupním souboru musí obsahovat vstup pro výsledek na prvním řádku výstupu.")
    print()

def generate_majority():
    input_size = input("Velikost vstupního vektoru (například 5 => 1, 2, 3, 1, 1): ")
    output_size = input("Počet povolených hodnot (například 5 => 0, 1, 2, 3, 4): ")
    train_size = input("Množství trénovacích dat: ")
    test_size = input("Množství testovacích dat: ")
    try:
        input_size = int(input_size)
        output_size = int(output_size)
        train_size = int(train_size)
        test_size = int(test_size)
    except:
        print("Některá z hodnot nebyla přirozené číslo")
        input("\nStiskněte enter pro pokračování...")
        return
    if input_size <= 0 or output_size <= 0 or train_size <= 0 or test_size <= 0:
        print("Některá z hodnot nebyla přirozené číslo")
        input("\nStiskněte enter pro pokračování...")
        return
    if input_size != current_network.input_size or output_size != current_network.output_size:
        print("Některá z hodnot vstupů či výstupů se neshoduje se současnou neuronovou sítí.")
        print("Přejete si načtenou síť zahodit?")
        input("\nStiskněte enter pro pokračování...")
        return
    
    global train_x, train_y, test_x, test_y
    train_x, train_y = tools.generate_majority(samples_number=train_size, input_size=input_size,
                                               output_size=output_size)
    test_x, test_y = tools.generate_majority(samples_number=train_size, input_size=input_size,
                                                output_size=output_size)

def learn():
    pass

def choose_task():
    print_choose_task(True)
    text = input("Vaše volba: ")
    if len(text) <= 0 or text[0] in ["h", "H"]:
        clear_screen()
        print_choose_task(False)
        help_choose()
        text = input("Vaše volba: ")
    if text <= 0 or text[0] not in ["m", "M", "o", "O", "x", "X", "c", "C"]:
        print("Neplatná volba.")
        input("\nStiskněte enter pro pokračování...")
        return
    elif text[0] in ["m", "M"]:
        pass






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
    global input_size, output_size
    input_size = current_network.input_size
    output_size = current_network.output_size

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

def learn():
    print(f"Kolik prvků z datasetu si přejete využít? (Maximum {train_x.shape[0]}.)")

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
    elif x[0] in ["l", "L"]:
        learn()
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
