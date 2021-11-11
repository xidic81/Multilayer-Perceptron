# -*- coding: utf-8 -*-


# Commented out IPython magic to ensure Python compatibility.
# %cd /content
!rm -r sample_data
!wget https://github.com/xidic81/Multilayer-Perceptron/blob/main/fashion_mnist_jpg.zip?raw=true -O fmnist.zip
!unzip fmnist.zip && rm fmnist.zip

"""This is some samples from dataset"""

import matplotlib.pyplot as plt
import numpy as np
import random
import os
from glob import glob

image_path_list = glob("/content/fashion_mnist_jpg/test/*/*.jpg")  

img1  = plt.imread(random.choice(image_path_list))
img2  = plt.imread(random.choice(image_path_list))
img3  = plt.imread(random.choice(image_path_list))
img4  = plt.imread(random.choice(image_path_list))


images = np.concatenate([img1,img2,img3,img4],1)
plt.gray()
plt.imshow(images)

def one_hot_encoding(n,n_classes):
  result = np.zeros(n_classes)
  result[n]=1.0
  return result

from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np

def load_dataset(root_path): 
  print(f"Memuat : {root_path}")
  categories = os.listdir(root_path)
  data = []
  for c in tqdm(categories,position=0,leave=True):
    c_dir = os.path.join(root_path,c)
    daftar_file_gambar = os.listdir(c_dir)
    for file_gambar in daftar_file_gambar:
      path_gambar = os.path.join(c_dir,file_gambar)
      data_gambar = plt.imread(path_gambar).flatten()
      c =np.array(c).reshape(1)
      item = np.concatenate([c,data_gambar])
      data.append(item) 
  return np.array(data,dtype=np.int64)

training_dataset = load_dataset(root_path = "/content/fashion_mnist_jpg/train")
validation_dataset = load_dataset(root_path = "/content/fashion_mnist_jpg/valid")
testing_dataset = load_dataset(root_path = "/content/fashion_mnist_jpg/test")

data = np.concatenate([training_dataset,validation_dataset,testing_dataset])
m, n = data.shape

np.random.shuffle(data) # shuffle before splitting into dev and training sets
print(data.shape)
data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape

def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
    
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(f"Accuracy : {get_accuracy(predictions, Y)*100.0}%")
    return W1, b1, W2, b2

W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 3000)
np.save("weight1.npy",W1)
np.save("bias1.npy",b1)
np.save("weight2.npy",W2)
np.save("bias2.npy",b2)

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

test_prediction(0, W1, b1, W2, b2)
test_prediction(1, W1, b1, W2, b2)
test_prediction(2, W1, b1, W2, b2)
test_prediction(3, W1, b1, W2, b2)

dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
get_accuracy(dev_predictions, Y_dev)
