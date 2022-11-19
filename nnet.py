import numpy as np
import mnist
from functions import *
import pickle
import matplotlib.pyplot as plt


class LAYER:
    def __init__(self, type, weights, biases, activation, activation_der):
        self.type = type
        if type == "FC":
            self.weights = weights
            self.biases = biases
        elif type == "Act":
            self.activation = activation
            self.activation_der = activation_der
    
    def __str__(self):
        if self.type == "FC":
            print("Váhy", self.weights)
            print("Biasy", self.biases)
            return ""
        elif self.type == "Act":
            print(self.activation(0))
            return ""
        else:
            return "Unknown"
    
    def forward_propagation(self, layer_input):
        if self.type == "FC":
            self.input_mem = layer_input
            self.output_mem = np.dot(layer_input, self.weights) + self.biases
            return self.output_mem
        elif self.type == "Act":
            self.input_mem = layer_input
            self.output_mem = self.activation(layer_input)
            return self.output_mem
        else:
            print("Dont know this layer")
    
    def backward_propagation(self, output_error, learning_rate):
        if self.type == "FC":
            input_error = np.dot(output_error, self.weights.T)
            weights_error = np.dot(self.input_mem.T, output_error)
            self.weights -= learning_rate * weights_error
            self.biases -= learning_rate * output_error
            return input_error
        elif self.type == "Act":
            return self.activation_der(self.input_mem) * output_error


class NNET:
    def load_nnet(file):
        with open(file, "rb") as inp:
            return pickle.load(inp)

    def save_nnet(self, file):
        with open(file, 'wb') as save:
            pickle.dump(self, save, pickle.HIGHEST_PROTOCOL)

    def __init__(self, input_size = 0, number_of_layers = 0, layer_types = [], layer_sizes = [], layer_ders = [], default_weights = None, 
                 default_biases = None, object_func = mse, object_func_der = mse_der, normalize_output = lambda x: x):
        self.number_of_layers = number_of_layers
        self.layer_sizes = layer_sizes
        self.layer_types = layer_types
        self.layers = []
        for i in range(number_of_layers):
            if default_weights != None:
                weights = default_weights[i]
                bias = default_biases[i]
            elif i == 0:
                weights = 2*np.random.rand(input_size, layer_sizes[i]) - 1
                bias = 2*np.random.rand(1, layer_sizes[i]) - 1
            elif i > 0:
                weights = 2*np.random.rand(layer_sizes[i - 1], layer_sizes[i]) - 1
                bias = 2*np.random.rand(1, layer_sizes[i]) - 1
            self.layers.append(LAYER("FC", weights, bias, None, None))
            self.layers.append(LAYER("Act", weights, bias, layer_types[i], layer_ders[i]))
        self.object_func = object_func
        self.object_func_der = object_func_der
    
    def forward_propagation(self, input):
        for layer in self.layers:
            input = layer.forward_propagation(input)
        return input

    def predict(self, input):
        output = self.forward_propagation(input)
        return np.argmax(soft_max(output)), soft_max(output)
        

    def backward_propagation(self, input, learning_rate):
        for layer in reversed(self.layers):
            input = layer.backward_propagation(input, learning_rate)
        return input

    def learn(self, inputs, labels, iterations, learning_rate):
        errors = []
        for i in range(iterations):
            error_val = 0
            for input, label in zip(inputs, labels):
                output = self.forward_propagation(input)
                # error = self.object_func_der(label, output)
                
                error = soft_max(output)
                error_val += cross_entropy(error, label)
                error = cross_entropy_der(error, label)
                self.backward_propagation(error, learning_rate)
            errors.append(error_val)

        plt.plot(errors)
        plt.savefig("img.png")


    

if __name__ == "__main__":
    network = NNET(input_size = 3, number_of_layers = 2, layer_types = [tanh, ReLU], layer_sizes = [10, 3], layer_ders = [tanh_der, ReLU_der])
    x_train = np.array([[[0,0, 0]], [[0,1, 0]], [[1,0, 1]], [[1,1, 1]], [[2, 2, 2]], [[1, 0, 0]], [[0, 0, 1]], [[1, 1, 0]]])
    y_train = np.array([[[1, 0, 0]], [[1, 0, 0]], [[0, 1, 0]], [[0, 1, 0]], [[0, 0, 1]], [[1, 0, 0]], [[1, 0, 0]], [[0, 1, 0]]])
    # network = NNET.load_nnet("file.pkl")
    network.learn(x_train, y_train, 1000, 0.1)
    network.predict([1, 0, 0])
    network.predict([0, 0, 0])
    network.predict([1, 1, 1])
    network.predict([2, 2, 2])
    network.save_nnet("file.pkl")
    # print(cross_entropy(network.forward_propagation(np.array([1, 0])), np.array([1, 0, 0, 0, 0])))
    pass