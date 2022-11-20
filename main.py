#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: David Hudak
# File: main.py
# Login: xhudak03
# Course: SFC
# School: BUT FIT
# Short description: This file contains UI implementation for project to course SFC

import os

import nnet
import tools

current_network = None

def print_loaded():
    print(f"Vítejte v programu pro práci s neuronovými sítěmi. \n\nAktuální stav globálních proměnných:")
    print(f"  Právě máte načtenou síť: {current_network}")

def print_menu():
    pass

def selection(x):
    if len(x) <= 0:
        return 1

if __name__ == "__main__":
    end = False
    while end == False:
        if os.name == "nt":
            os.system("CLS")
        else:
            os.system("clear")
        print_loaded()
        text = input("Vaše volba ")
        if len(text) == 0 or text[0] not in ["Y", "y"]:
            end = True
