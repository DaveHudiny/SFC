from nnet import *

# training data
x_train = np.array([[[0,0, 0]], [[0,1, 0]], [[1,0, 1]], [[1,1, 1]], [[2, 2, 2]], [[1, 0, 0]]], )
y_train = np.array([[[1, 0, 0]], [[1, 0, 0]], [[0, 1, 0]], [[0, 1, 0]], [[0, 0, 1]], [[1, 0, 0]]])

x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[1]]])

# network
net = Network()
net.add(FCLayer(3, 3))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(3, 3))
net.add(ActivationLayer(tanh, tanh_prime))

# train
net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs=1000, learning_rate=0.1)

# test
out = net.predict(x_train)
print(out)