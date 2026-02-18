import numpy as np
import pickle
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(
        normalize=True, flatten=True, one_hot_label=False
    )
    return x_test, t_test


def init_network():
    with open(os.path.dirname(__file__) + "/sample_weight.pkl", "rb") as f:
        network = pickle.load(f)
        print(network)
        input("Press Enter to continue...")
    return network


def predict(network, x):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    a1 = np.dot(x, W1) + b1
    print("x:", x.shape)
    print("W1:", W1.shape)
    print("b1:", b1.shape)
    print("a1:", a1.shape)
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    print("z1:", z1.shape)
    print("W2:", W2.shape)
    print("b2:", b2.shape)
    print("a2:", a2.shape)
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    print("z2:", z2.shape)
    print("W3:", W3.shape)
    print("b3:", b3.shape)
    print("a3:", a3.shape)
    print("y:", y.shape)

    return y


if __name__ == "__main__":
    x, t = get_data()
    network = init_network()
    accuracy_cnt = 0
    for i in range(len(x)):
        y = predict(network, x[i])
        p = np.argmax(y)
        if p == t[i]:
            accuracy_cnt += 1

    print("Accuracy: " + str(float(accuracy_cnt) / len(x)))
