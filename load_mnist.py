from keras.datasets import mnist
import pickle 
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

with open("train_x.pkl", 'wb') as save:
  pickle.dump(x_train, save, pickle.HIGHEST_PROTOCOL)
with open("train_y.pkl", 'wb') as save:
  pickle.dump(y_train, save, pickle.HIGHEST_PROTOCOL)
with open("test_x.pkl", 'wb') as save:
  pickle.dump(x_test, save, pickle.HIGHEST_PROTOCOL)
with open("test_y.pkl", 'wb') as save:
  pickle.dump(y_test, save, pickle.HIGHEST_PROTOCOL)