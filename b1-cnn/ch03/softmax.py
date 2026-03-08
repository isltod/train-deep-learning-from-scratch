import numpy as np


def softmax_overflow(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    print(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


if __name__ == "__main__":
    a = np.array([1010, 1000, 990])
    y = softmax(a)
    print(y)
    b = np.array([100, 1000, 2000])
    y = softmax(b)
    print(y)
