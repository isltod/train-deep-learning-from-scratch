import numpy as np


def softmax(x):
    """소프트맥스 함수"""
    e_x = np.exp(x - np.max(x))  # 수치 안정성을 위해 max 빼줌
    return e_x / e_x.sum()


def softmax_derivative(s):
    """소프트맥스 미분 (Jacobian Matrix)"""
    # s는 softmax의 출력 (1 x n)
    s = s.reshape(-1, 1)
    # diag(s) - s * s.T
    return np.diagflat(s) - np.dot(s, s.T)


# 사용 예시
x = np.array([1.0, 2.0, 3.0])
s = softmax(x)
derivative = softmax_derivative(s)
print("Softmax Output:\n", s)
print("\nSoftmax Derivative Matrix:\n", derivative)
