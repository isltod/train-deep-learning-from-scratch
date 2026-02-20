import numpy as np


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    # 다차원 배열 이터레이터, flags=['multi_index'] -> 다차원 인덱스(예: (0, 0), (0, 1))
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index  # 현재 요소의 다차원 인덱스
        tmp_val = x[idx]  # 원래 값 임시 저장
        print(idx, tmp_val)

        # f(x+h) 계산
        # 여기 x는 simpleNet 클래스의 W를 참조로 받아왔으니, 여기서 고치면 그 값이 변한다...
        x[idx] = float(tmp_val) + h
        # 매개변수로 simpleNet.W = x 를 넘기지만 이건 더미 변수로 아무 일도 안한다. 그래서 헷갈린다...
        # f는 simpleNet.loss로 손실함수이고 입력치와 정답지는 람다함수 선언에서 이미 고정된채로 f와 묶여있고,
        # 입력치에 가중치를 곱해서 예측치를 만들어 정답지와 비교해야 하는데,
        # 가중치는 위에서 고친걸로 이미 반영되서 여기서 넘길 필요가 없다...왜 이렇게 헷갈리게 코드를 만들었을까...
        fxh1 = f(x)

        # f(x-h) 계산
        x[idx] = tmp_val - h
        fxh2 = f(x)

        # 중심 차분(central difference)으로 기울기 계산
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val  # 값 복원
        it.iternext()  # 다음 요소로 이동

    return grad
