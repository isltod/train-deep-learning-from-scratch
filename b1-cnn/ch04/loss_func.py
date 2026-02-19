import numpy as np


def sum_squares_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error(y, t):
    # 배치로 들어올 때는 (n, 10)으로 들어오는데, 이미지 하나씩 처리할 때는 (10,)으로 들어온다.
    # 이걸 맞춰주려면 이미지 하나씩 들어올 때 (1, 10)으로 바꿔줘야 한다...
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    delta = 1e-7
    return -np.sum(t * np.log(y + delta)) / batch_size
    # 원핫인코딩이 아닐 때는 정답이 [2, 7, 0, 9, 4]처럼 들어오므로,
    # y[0, 2], y[1, 7]...등이 t와 비교되어야 한다...그래서
    # return -np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size
    # y[[0, 1, 2, 3, 4], [2, 7, 0, 9, 4]] 꼴로...


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
