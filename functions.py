import numpy as np

def count_success(network, test_x, test_y):
    successful = 0
    for x, y in zip(test_x, test_y):
        y_pred, _ = network.predict(x)
        if y_pred == np.argmax(y):
            successful += 1
    print("Successful was", successful, "from", len(test_y))
    print("Success rate:", successful/len(test_y)*100, "%")

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

def mse(output, expected_output):
    return np.mean(np.power(expected_output-output, 2))

def mse_der(output, expected_output):
    return 2*(output-expected_output)/expected_output.size

def cross_entropy(output, expected_output):
    log_likelihood = expected_output * np.log(output)
    return - np.sum(log_likelihood)

def cross_entropy_der(output, expected_output):
    return output - expected_output

def soft_max(input):
    return np.exp(input) / (np.sum(np.exp(input)) + 1e-10)

def linear(input):
    return input

def linear_der(input):
    return 1