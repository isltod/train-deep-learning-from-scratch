import numpy as np


def sum_squares_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


if __name__ == "__main__":
    # 이게 맞는 답이고
    t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

    # 이렇게 예측하면 SSE 0.0975, CEE 0.51
    y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
    print("SSE O:", sum_squares_error(np.array(y), np.array(t)))
    print("CEE O:", cross_entropy_error(np.array(y), np.array(t)))

    # 이렇게 예측하면 0.5975, 2.30
    y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
    print("SSE X:", sum_squares_error(np.array(y), np.array(t)))
    print("CEE X:", cross_entropy_error(np.array(y), np.array(t)))
