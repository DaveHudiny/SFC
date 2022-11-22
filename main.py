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

train_x, train_y, test_x, test_y = None, None, None, None


def print_loaded():
    print(f"Vítejte v programu pro práci s neuronovými sítěmi. \n\nAktuální stav globálních proměnných:")
    print(f"  Právě máte načtenou síť: {current_network}")
    print(
        f"  Grafy vývoje chybové funkce se aktuálně nachází v souboru: {loss_img}")
    print(
        f"  Počet aktuálních vstupů sítě/úlohy: {input_size} (pokud -1, vstup nebyl určen)")
    print(
        f"  Počet aktuálních výstupů sítě/úlohy: {output_size} (pokud -1, výstup nebyl určen)")
    print(f"  Aktuální úloha {task}")


def print_menu():
    print("Co si přejete dělat?")
    print("E) Nauč síť na trénovacích datech")
    print("P) Vyzkoušet síť na testovacích datech")
    print("T) Vybrat úlohu")
    print("C) Vytvořit vlastní síť")
    print("L) Načíst síť")
    print("S) Uložit síť")

    print("Q) Ukončit program")
    print()
    pass


def print_choose_task():
    print("Vyberte jednu z následujících úloh:")
    print("  m) MNIST")
    print("  o) Výpočet modu (nejčastější hodnoty)")
    print("  x) XOR problém")
    print("  c) Vlastní problém")
    print()


def help_choose():
    print("V případě volby vlastního formátu musíte mít alespoň jeden soubor se vstupy (pokud trénovací data == testovací data) a jeden s výstupy.")
    print("Soubor se vstupy musí obsahovat posloupnost desetinných čísel od -1.0 do 1.0 oddělených mezerami, kde řádek znamená jeden vstupní vektor.")
    print("Soubor s výstupy musí obsahovat posloupnost desetinných čísel od 0.0 do 1.0 (příslušnosti) oddělených mezerami, kde řádek symbolizuje vektor výstupu pro vstup.")
    print("Desátý řádek ve vstupním souboru musí obsahovat vstup pro výsledek na desátém řádku výstupu.")
    print("Záznamy všech musí mít stejné délky. Záznamy všech výstupů rovněž. Kontrola nebude z důvodu náročnosti prováděna.")
    print()


def generate_majority():
    input_size_l = input(
        "Velikost vstupního vektoru (například 5 => 1, 2, 3, 1, 1): ")
    output_size_l = input(
        "Počet povolených hodnot (například 5 => 0, 1, 2, 3, 4): ")
    train_size = input("Množství trénovacích dat: ")
    test_size = input("Množství testovacích dat: ")
    global current_network
    try:
        input_size_l = int(input_size_l)
        output_size_l = int(output_size_l)
        train_size = int(train_size)
        test_size = int(test_size)
    except:
        print("Některá z hodnot nebyla přirozené číslo")
        input("\nStiskněte enter pro pokračování...")
        return
    if input_size_l <= 0 or output_size_l <= 0 or train_size <= 0 or test_size <= 0:
        print("Některá z hodnot nebyla přirozené číslo")
        input("\nStiskněte enter pro pokračování...")
        return
    if current_network != None:
        if input_size_l != current_network.input_size or output_size_l != current_network.output_size:
            print(
                "Některá z hodnot vstupů či výstupů se neshoduje se současnou neuronovou sítí.")
            exitor = input("Přejete si načtenou síť zahodit? Vaše volba (y, Y):")
            if len(exitor) > 0 and exitor[0] in ["Y", "y", "a", "A"]:
                exitor = True
            else:
                exitor = False
            if not exitor:
                input("\nStiskněte enter pro pokračování...")
                return
    current_network = None
    global input_size, output_size
    input_size = input_size_l
    output_size = output_size_l
    global train_x, train_y, test_x, test_y
    train_x, train_y = tools.generate_majority(samples_number=train_size, input_size=input_size,
                                               output_size=output_size)
    test_x, test_y = tools.generate_majority(samples_number=test_size, input_size=input_size,
                                             output_size=output_size)
    global task
    train_x = train_x / float(output_size)
    test_x = test_x / float(output_size)
    train_x -= 0.5
    test_x -= 0.5
    task = "Majority"


def load_mnist():
    global task
    task = "MNIST"
    global train_x, train_y, test_x, test_y
    train_x, train_y, test_x, test_y = tools.load_mnist()
    global input_size, output_size
    train_x = train_x.reshape(train_x.shape[0], 1, 28*28)
    test_x = test_x.reshape(test_x.shape[0], 1, 28*28)
    input_size = 784
    output_size = 10
    train_x = train_x.astype('float32')
    train_x = train_x / 255.0
    train_x -= 0.5
    test_x = test_x.astype('float32')
    test_x = test_x / 255.0
    test_x -= 0.5

def load_custom_task():
    global train_x, train_y, test_x, test_y
    global input_size, output_size
    clear_screen()
    help_choose()
    print()
    f_trainx = input("Soubor s trénovacími vstupy: ")
    if not os.path.isfile(f_trainx):
        print("Zadaný soubor se nezdařilo načíst")
        input("\nStiskněte enter pro pokračování...")
        return
    f_trainy = input("Soubor s trénovacími výstupy: ")
    if not os.path.isfile(f_trainy):
        print("Zadaný soubor se nezdařilo načíst")
        input("\nStiskněte enter pro pokračování...")
        return
    f_testx = input("Soubor s testovacími vstupy: ")
    if not os.path.isfile(f_testx):
        print("Zadaný soubor se nezdařilo načíst")
        input("\nStiskněte enter pro pokračování...")
        return
    f_testy = input("Soubor s testovacími výstupy: ")
    if not os.path.isfile(f_testy):
        print("Zadaný soubor se nezdařilo načíst")
        input("\nStiskněte enter pro pokračování...")
        return
    try:
        train_x, train_y, test_x, test_y = tools.load_custom_task(f_trainx, f_trainy, f_testx, f_testy)
    except KeyboardInterrupt:
        print("Program byl násilně ukončen při čtení souborů!")
        exit()
    except:
        print("Soubory nejsou správného formátu.")
        input("\nStiskněte enter pro pokračování...")
        return
    input_size = len(train_x[0][0])
    output_size = len(train_y[0][0])
    input("\nStiskněte enter pro pokračování...")

def choose_task():
    print_choose_task()
    text = input("Vaše volba: ")
    if len(text) <= 0 or text[0] in ["h", "H"]:
        clear_screen()
        print_choose_task()
        help_choose()
        text = input("Vaše volba: ")
    if len(text) <= 0 or text[0] not in ["m", "M", "o", "O", "x", "X", "c", "C"]:
        print("Neplatná volba.")
        input("\nStiskněte enter pro pokračování...")
        return
    elif text[0] in ["m", "M"]:
        load_mnist()
    elif text[0] in ["o", "O"]:
        generate_majority()
    elif text[0] in ["c", "C"]:
        load_custom_task()


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
    global current_network
    if train_x is None or train_y is None or current_network is None:
        print("Nemáte načtenou síť nebo trénovací data.")
        input("\nStiskněte enter pro pokračování...")
    amount = input(
        f"Kolik záznamů z datasetu si přejete využít? (Maximum {train_x.shape[0]}): ")
    learning_rate = input(f"Koeficient učení: ")
    iterations = input("Počet epoch: ")
    often = input(
        "Jak často chcete tisknout chybu (tiskne se každá epocha mod often == 0): ")
    split = input("Přejete si trénovací data rozdělit do podskupin (Y/n): ")
    try:
        amount = int(amount)
        learning_rate = float(learning_rate)
        iterations = int(iterations)
        often = int(often)
        if amount <= 0 or learning_rate <= 0 or iterations <= 0 or often <= 0:
            assert()
        if amount >= train_x.shape[0]:
            amount = train_x.shape[0]
    except:
        print("Zadána ilegální hodnota některého parametru.")
        input("\nStiskněte enter pro pokračování...")
        return
    if len(split) > 0 and split[0] in ["Y", "y", "a", "A"]:
        split = True
    else:
        split = False
    if split:
        how_split = input("Počet částí původního trénovacího vzoru: ")
        metapochs = input("Počet iterací nad všemi částmi: ")
        try:
            how_split = int(how_split)
            metapochs = int(metapochs)
        except:
            print("Zadána ilegální hodnota některého parametru.")
            input("\nStiskněte enter pro pokračování...")
            return
    else:
        how_split = 0
        metapochs = 0
    tools.learn(task, train_x[:amount], train_y[:amount], learning_rate, loss_img, current_network,
                iterations, often, split, how_split, metapochs)


def predict():
    global test_x, test_y, current_network

    if test_x is None or test_y is None or current_network is None:
        print("Musí být načtena jak testovací data, tak neuronová síť.")
        input("\nStiskněte enter pro pokračování...")
        return
    count_success(current_network, test_x, test_y)
    input("\nStiskněte enter pro pokračování...")


def selection(x):
    if len(x) == 0:
        return 0
    elif x[0] in ["q", "Q"]:
        return 1
    elif x[0] in ["l", "L"]:
        load_network()
    elif x[0] in ["s", "S"]:
        save_network()
    elif x[0] in ["c", "C"]:
        global current_network, input_size, output_size
        current_network = cn.create_network(input_size, output_size)
        if current_network != None:
            input_size = current_network.input_size
            output_size = current_network.output_size
    elif x[0] in ["t", "T"]:
        choose_task()
    elif x[0] in ["e", "E"]:
        learn()
    elif x[0] in ["p", "P"]:
        predict()
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
