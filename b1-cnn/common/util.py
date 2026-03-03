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


def shuffle_dataset(x, t):
    """데이터셋을 뒤섞는다"""
    # 주어진 숫자 내의 정수로 순열을 만든다
    permutation = np.random.permutation(x.shape[0])
    # 어쨌든 첫 번째 차원에 대해서만 뒤섞으면 되는 건가? 뒤는 원핫인코딩, 배치, 뭐 그런건가?
    x = x[permutation, :] if x.ndim == 2 else x[permutation, :, :, :]
    t = t[permutation]

    return x, t


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """다수의 이미지를 입력받아 2차원 배열로 변환한다(평탄화).

    Parameters
    ----------
    input_data : 4차원 배열 형태의 입력 데이터(이미지 수, 채널 수, 높이, 너비)
    filter_h : 필터의 높이
    filter_w : 필터의 너비
    stride :
    pad :

    Returns
    -------
    col : 2차원 배열
    """
    N, C, H, W = input_data.shape
    # //는 나누기 몫 - 나눗셈 정수 부분
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    # input_data에 패딩 넣기, (0, 0)은 각 차원별로 (좌,우) 형식, 값은 constant - 일정한 값? 뭐지?
    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], "constant")
    # 6차원(N, C, FH, FW, OH, OW) 0행렬 만들기
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    # 필터 세로/가로 크기대로 돌면서
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            # 모든 배치의 모든 채널에 대해, 0번째 위치부터 필터 크기까지를, 각 위치 인덱스에 묶어 담기...
            # 예를 들어 0: 0~2, 1: 1~3, 2: 2~4, ...
            # y:y_max:stride - y부터 y_max 전까지 stride로 건너뛰면서
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    # 3차원 이상 텐서에서 축 순서를 지정해서 전치 - 필터에 적용될 이미지 세로, 가로를 앞으로, 다음 채널, 그리고 필터 위치
    # 그리고는 2차원으로 다 묶는다... 행은 배치 갯수 x 출력 세로 x 출력 가로, 나머지는 다 열로(필터 크기, 채널 수)
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """(im2col과 반대) 2차원 배열을 입력받아 다수의 이미지 묶음으로 변환한다.

    Parameters
    ----------
    col :
    input_shape : 원래 이미지 데이터의 형상（예：(10, 1, 28, 28)）
    filter_h :
    filter_w
    stride
    pad

    Returns
    -------
    imgs : 변환된 이미지들
    """
    # 원래 이미지 shape 받아놓고,
    N, C, H, W = input_shape
    # im2col에서 작업한 out 세로 가로 계산해서 im2col로 작업한 shape로 col 변경
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(
        0, 3, 4, 5, 1, 2
    )

    # 좀 더 남게 원래 이미지를 0행렬로 만든다..왜 남게 하지?
    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]
    # 그리고는 원래 이미지 shape대로 잘라서 반환한다...패딩 효과가 뭔가 있나?
    return img[:, :, pad : H + pad, pad : W + pad]


if __name__ == "__main__":
    a = np.array([1, 2, 3, 4, 5])
    lw = 3
    d = np.r_[a[lw - 1 : 0 : -1], a, a[-1:-lw:-1]]
    print(a[lw - 1 : 0 : -1])
    print(a[-1:-lw:-1])
    print(d.shape)
    print(d)
