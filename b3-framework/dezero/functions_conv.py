import numpy as np
from dezero import cuda
from dezero.core import Function, as_variable
from dezero.functions import linear, broadcast_to
from dezero.utils import get_conv_outsize, get_deconv_outsize, pair


def _im2col(img, kernel_size, stride, pad, to_matrix=True):
    n, c, h, w = img.shape
    kh, kw = pair(kernel_size)
    sy, sx = pair(stride)
    ph, pw = pair(pad)
    out_h = get_conv_outsize(h, kh, sy, ph)
    out_w = get_conv_outsize(w, kw, sx, pw)
    dy, dx = 1, 1
    col = cuda.cupy.empty((n, c, kh, kw, out_h, out_w), dtype=img.dtype)

    cuda.cupy.ElementwiseKernel(
        # 이게 입력 매개변수들, raw는 수동 인덱싱 사용으로 아래 i나 [] 안의 변수 사용...
        "raw T img, int32 h, int32 w, int32 out_h, int32 out_w,"
        "int32 kh, int32 kw, int32 sy, int32 sx, int32 ph, int32 pw,"
        "int32 dy, int32 dx",
        # 이건 출력 변수, T는 일반 타입? 안 정해진 타입이라는 듯...
        "T col",
        # 이건 메인 루프문...뭘 하는 건지 모르겠네...
        """
           int c0 = i / (kh * kw * out_h * out_w);
           int ky = i / (kw * out_h * out_w) % kh;
           int kx = i / (out_h * out_w) % kw;
           int out_y = i / out_w % out_h;
           int out_x = i % out_w;
           int in_y = ky * dy + out_y * sy - ph;
           int in_x = kx * dx + out_x * sx - pw;
           if (in_y >= 0 && in_y < h && in_x >= 0 && in_x < w) {
             col = img[in_x + w * (in_y + h * c0)];
           } else {
             col = 0;
           }
        """,
        # 커널 이름이라...
        "im2col",
        # reduced_view는 크기 1인 차원 제거
    )(img.reduced_view(), h, w, out_h, out_w, kh, kw, sy, sx, ph, pw, dy, dx, col)

    return col


def im2col_array(img, kernel_size, stride, pad, to_matrix=True):
    # 배치, 채널, 세로, 가로
    N, C, H, W = img.shape
    # 커널, 스트라이드, 패드 크기 받아서 출력 크기 결정
    KH, KW = pair(kernel_size)
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    xp = cuda.get_array_module(img)
    if xp != np:
        # 쿠파이 버전이면 _im2col로 보내는데, 이게 뭘 하는건지 모르겠다...
        col = _im2col(img, kernel_size, stride, pad)
    else:
        # 아니면 넘파이에서 해결하는데...
        # 패딩 넣기, 인수는 원본 배열, 패딩 크기, 값
        # 패딩 크기는 각 차원별로 (시작,끝) 형식 - 배치, 채널은 패딩 없고, 가로 세로는 스트라이드 고려해서 넣기
        # 값은 constant - 기본 0
        img = np.pad(
            img, ((0, 0), (0, 0), (PH, PH + SH - 1), (PW, PW + SW - 1)), "constant"
        )
        # 빈 배열 만들고
        col = np.ndarray((N, C, KH, KW, OH, OW), dtype=img.dtype)
        # 필터 세로/가로 크기대로 돌면서
        for y in range(KH):
            y_max = y + SH * OH
            for x in range(KW):
                x_max = x + SW * OW
                # 모든 배치의 모든 채널에 대해, 0번째 위치부터 필터 크기까지를, 각 위치 인덱스에 묶어 담기...
                # 예를 들어 0: 0~2, 1: 1~3, 2: 2~4, ...
                # y:y_max:stride -> y부터 y_max 전까지 stride로 건너뛰면서
                col[:, :, y, x, :, :] = img[:, :, y:y_max:SH, x:x_max:SW]

    # 행렬 형태로 반환하라고 했으면...
    # 3차원 이상 텐서에서 축 순서를 지정해서 전치 - 필터에 적용될 이미지 세로, 가로를 앞으로, 다음 채널, 그리고 필터 위치
    # 그리고는 2차원으로 다 묶는다... 행은 배치 갯수 x 출력 세로 x 출력 가로, 나머지는 다 열로(필터 크기, 채널 수)
    if to_matrix:
        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * OH * OW, -1)
    return col


class Im2col(Function):
    def __init__(self, kernel_size, stride, pad, to_matrix=True):
        self.input_shape = None
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.to_matrix = to_matrix

    def forward(self, x):
        self.input_shape = x.shape
        xp = cuda.get_array_module(x)
        self.x_shape = x.shape


def im2col(x, kernel_size, stride, pad, to_matrix=True):
    return Im2col(kernel_size, stride, pad, to_matrix)(x)


class Col2im(Function):
    def __init__(self, kernel_size, stride, pad):
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        xp = cuda.get_array_module(x)
        self.x_shape = x.shape


def col2im(x, kernel_size, stride, pad):
    return Col2im(kernel_size, stride, pad)(x)
