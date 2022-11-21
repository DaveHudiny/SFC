from nnet import NNET
import tools
from functions import *

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
    current_network = NNET(input_size, layer_types, layer_sizes, layer_types_der, None, None, loss_fun, loss_fun_der, 1, name)
    x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
    y_train = np.array([[[1, 0]], [[0, 1]], [[0, 1]], [[1, 0]]])
    current_network.learn(x_train, y_train, 1000, 0.1)
    tools.count_success(current_network, x_train, y_train)
    print(f"Úspěšně jste vytvořil neuronovou síť {name}")
    print()
    input("Stiskněte enter pro pokračování...")
    return current_network