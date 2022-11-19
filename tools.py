import numpy as np
from nnet import *
import pickle

def generate_majority(samples_number = 200, input_size = 5, output_size = 3):
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

def count_success(network, test_x, test_y):
    successful = 0
    for x, y in zip(test_x, test_y):
        y_pred, _ = network.predict(x)
        if y_pred == np.argmax(y):
            successful += 1
    print("Successful was", successful, "from", len(test_y))
    print("Success rate:", successful/len(test_y)*100, "%")

def load_mnist():
    with open("./train/train_mnist/train_x.pkl", "rb") as inp:
        train_x = pickle.load(inp)
    with open("./train/train_mnist/train_y.pkl", "rb") as inp:
        train_y = pickle.load(inp)
    with open("./test/test_mnist/test_x.pkl", "rb") as inp:
        test_x = pickle.load(inp)
    with open("./test/test_mnist/test_y.pkl", "rb") as inp:
        test_y = pickle.load(inp)
    network = NNET(input_size = 784, number_of_layers = 4, layer_types = [tanh, tanh, tanh, tanh], layer_sizes=[300, 300, 300, 10], layer_ders=[tanh_der, tanh_der, tanh_der, tanh_der])
    network.learn(train_x[0:700], train_y[0:700], 50, learning_rate=0.1)
    count_success(network, test_x[0:50], test_y[0:50])

    

    return
    network = NNET(input_size = 784, number_of_layers = 3, layer_types = [tanh, tanh, tanh], layer_sizes = [100, 100, 10], layer_ders = [tanh_der, tanh_der, tanh_der])
    network.learn(train_x, train_y)


load_mnist()
exit()
train_x, train_y = generate_majority(samples_number=500, output_size=4)
test_x, test_y = generate_majority(samples_number=50, output_size=4)

network = NNET(input_size = 5, number_of_layers = 3, layer_types = [tanh, tanh, tanh], layer_sizes = [100, 100, 4], layer_ders = [tanh_der, tanh_der, tanh_der])
network.learn(train_x, train_y, 1000, 0.01)
count_success(network, test_x, test_y)
