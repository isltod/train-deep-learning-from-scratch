import numpy as np


# f는 함수, x는 함수의 인수로 NumPy 배열
def numerical_gradient(f, x):
    h = 1e-4  # 미분에 사용할 아주 작은 값
    grad = np.zeros_like(
        x
    )  # x와 형상이 같은 배열을 생성하고 0으로 채움 (기울기를 저장할 배열)

    # np.nditer는 다차원 배열을 위한 이터레이터입니다.
    # flags=['multi_index']는 다차원 인덱스를 사용할 수 있게 합니다. (예: (0, 0), (0, 1))
    # op_flags=['readwrite']는 배열의 요소를 읽고 쓸 수 있게 합니다.
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index  # 현재 요소의 다차원 인덱스
        tmp_val = x[idx]  # 원래 값 임시 저장

        # f(x+h) 계산
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)

        # f(x-h) 계산
        x[idx] = tmp_val - h
        fxh2 = f(x)

        # 중심 차분(central difference)으로 기울기 계산
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val  # 값 복원
        it.iternext()  # 다음 요소로 이동

    return grad
