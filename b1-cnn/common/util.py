import numpy as np


def smooth_curve(x):
    """손실 함수의 그래프를 매끄럽게 하기 위해 사용"""
    window_length = 11
    # 1. x를 11부터 0 전, 즉 1까지 -1씩 줄이며 앞으로 뽑고(10개),
    # 2. 가운데는 원래 벡터,
    # 3. 마지막 부터 11개 전의 앞, 즉 10개 전까지 역순으로 뽑고
    # 이렇게 세 개를 행으로 붙인다? 이게 뭐지?
    s = np.r_[x[window_length - 1 : 0 : -1], x, x[-1:-window_length:-1]]
    # 뭔가 양 끝을 줄여줘서 스무딩 효과를 주는 모양...
    w = np.kaiser(window_length, 2)
    # 합성곱이라는데, 필터를 곱해서 합해서 한 점에 맵핑한다는 얘기긴 한데...
    y = np.convolve(w / w.sum(), s, mode="valid")
    return y[5 : len(y) - 5]


if __name__ == "__main__":
    a = np.array([1, 2, 3, 4, 5])
    lw = 3
    d = np.r_[a[lw - 1 : 0 : -1], a, a[-1:-lw:-1]]
    print(a[lw - 1 : 0 : -1])
    print(a[-1:-lw:-1])
    print(d.shape)
    print(d)
