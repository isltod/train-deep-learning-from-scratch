import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def identity_function(x):
    return x


# 0층
X = np.array([1.0, 0.5])
print("X:", X.shape)

# 첫 번째 가중치와 편향
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])
print("W1:", W1.shape)
print("B1:", B1.shape)

# 1층 - 0층의 전달 값과 활성화 함수 통과 값
A1 = np.dot(X, W1) + B1
print("XW1 + B1 = A1:", A1)

Z1 = sigmoid(A1)
print("Z1:", Z1)

# 두 번째 가충치와 편향
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])
print("Z1:", Z1.shape)
print("W2:", W2.shape)
print("B2:", B2.shape)

# 2층 - 1층의 전달 값과 활성화 함수 통과 값
A2 = np.dot(Z1, W2) + B2
print("XW2 + B2 = A2:", A2)

Z2 = sigmoid(A2)
print("Z2:", Z2)

# 세 번째 가중치와 편향
W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])
print("Z2:", Z2.shape)
print("W3:", W3.shape)
print("B3:", B3.shape)

# 3층 - 2층의 전달 값과 활성화 함수 통과 값
A3 = np.dot(Z2, W3) + B3
print("XW3 + B3 = A3:", A3)

Y = identity_function(A3)
print("Y:", Y)
