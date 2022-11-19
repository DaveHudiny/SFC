import numpy as np

def tanh_prime(x):
    return 1-np.tanh(x)**2

def tanh(x):
    return np.tanh(x)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_der(x):
    y = sigmoid(x)
    return y * (1 - y)

def tanh_der(x):
    return 1-np.tanh(x)**2

def ReLU(x):
    return np.maximum(0, x)

def ReLU_der(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x

def softmax_der(x):
    I = np.eye(x.shape[0])
    return soft_max(x) * (I - soft_max(x).T)

def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size

def mse_der(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size

def cross_entropy(output, expected_output):
    log_likelihood = expected_output * np.log(output)
    return - np.sum(log_likelihood)

def cross_entropy_der(output, expected_output):
    return output - expected_output

def soft_max(input):
    return np.exp(input) / (np.sum(np.exp(input)) + 1e-10)