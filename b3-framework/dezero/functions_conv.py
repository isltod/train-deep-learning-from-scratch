import numpy as np
from dezero import cuda
from dezero.core import Function, as_variable
from dezero.functions import linear, broadcast_to
from dezero.utils import get_conv_outsize, get_deconv_outsize, pair


# _im2col_gpu/_col2im_gpu, im2col_array/col2im_array 네 개가 실제 핵심 함수고,
# im2col_array/col2im_array가 책임지고 gpu 연산의 경우는 _im2col_gpu/_col2im_gpu로 보내는 로직
# im2col은 입력 데이터를 배치크기 x 커널크기로 반복 리셈플링,
# col2im은 반대로 배치크기 x 커널크기 리샘플링 결과를 원래 배치x채널x세로x가로 데이터로 되돌리기...
# 이 경우 forward는 im2col, backward 미분은 반대 col2im이고,
# im2col은 분기라서 col2im은 선택이 아니고 더해 넣는다...
def _im2col_gpu(img, kernel_size, stride, pad, to_matrix=True):
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


def _col2im_gpu(col, sy, sx, ph, pw, h, w):
    n, c, kh, kw, out_h, out_w = col.shape
    dy, dx = 1, 1
    img = cuda.cupy.empty((n, c, h, w), dtype=col.dtype)

    cuda.cupy.ElementwiseKernel(
        "raw T col, int32 h, int32 w, int32 out_h, int32 out_w,"
        "int32 kh, int32 kw, int32 sy, int32 sx, int32 ph, int32 pw,"
        "int32 dx, int32 dy",
        "T img",
        """
           int c0 = i / (h * w);
           int y  = i / w % h;
           int x  = i % w;
           T val = 0;
           for (int ky = 0; ky < kh; ++ky) {
             int out_y = (y + ph - ky * dy);
             if (0 > out_y || out_y >= out_h * sy) continue;
             if (out_y % sy != 0) continue;
             out_y /= sy;
             for (int kx = 0; kx < kw; ++kx) {
               int out_x = (x + pw - kx * dx);
               if (0 > out_x || out_x >= out_w * sx) continue;
               if (out_x % sx != 0) continue;
               out_x /= sx;
               int k = out_y + out_h * (kx + kw * (ky + kh * c0));
               val = val + col[out_x + out_w * k];
             }
           }
           img = val;
        """,
        "col2im",
    )(col.reduced_view(), h, w, out_h, out_w, kh, kw, sy, sx, ph, pw, dx, dy, img)
    return img


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
        col = _im2col_gpu(img, kernel_size, stride, pad)
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


def col2im_array(col, img_shape, kernel_size, stride, pad, to_matrix=True):
    # 배치, 채널, 세로, 가로
    N, C, H, W = img_shape
    # 커널, 스트라이드, 패드 크기 받아서 출력 크기 결정
    KH, KW = pair(kernel_size)
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    # 행렬 형태로 반환하라고 했으면...
    # 3차원 이상 텐서에서 축 순서를 지정해서 전치 - 필터에 적용될 이미지 세로, 가로를 앞으로, 다음 채널, 그리고 필터 위치
    # 그리고는 2차원으로 다 묶는다... 행은 배치 갯수 x 출력 세로 x 출력 가로, 나머지는 다 열로(필터 크기, 채널 수)
    if to_matrix:
        col = col.reshape(N, OH, OW, C, KH, KW).transpose(0, 3, 4, 5, 1, 2)

    xp = cuda.get_array_module(col)
    if xp != np:
        # 쿠파이 버전이면 _im2col로 보내는데, 이게 뭘 하는건지 모르겠다...
        img = _col2im_gpu(col, SH, SW, PH, PW, H, W)
        return img
    else:
        # 아니면 넘파이에서 해결하는데...좀 더 남게 원래 이미지를 0행렬로 만든다..왜 남게 하지?
        img = np.zeros(
            (N, C, H + 2 * PH + SH - 1, W + 2 * PW + SW - 1), dtype=col.dtype
        )
        for j in range(KH):
            j_lim = j + SH * OH
            for i in range(KW):
                i_lim = i + SW * OW
                # 모든 배치의 모든 채널에 대해, 0번째 위치부터 필터 크기까지를, 각 위치 인덱스에 더한다...
                # forward 분기의 역전파니까 더한다...
                # y:y_max:stride -> y부터 y_max 전까지 stride로 건너뛰면서
                img[:, :, j:j_lim:SH, i:i_lim:SW] += col[:, :, j, i, :, :]
        # 그리고는 원래 이미지 shape대로 잘라서 반환한다...패딩 효과가 뭔가 있나?
        return img[:, :, PH : H + PH, PW : W + PW]


class Im2col(Function):
    def __init__(self, kernel_size, stride, pad, to_matrix=True):
        super().__init__()
        self.input_shape = None
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.to_matrix = to_matrix

    def forward(self, x):
        self.input_shape = x.shape
        y = im2col_array(x, self.kernel_size, self.stride, self.pad, self.to_matrix)
        return y

    def backward(self, gy):
        gx = col2im(
            gy,
            self.input_shape,
            self.kernel_size,
            self.stride,
            self.pad,
            self.to_matrix,
        )
        return gx


def im2col(x, kernel_size, stride, pad, to_matrix=True):
    return Im2col(kernel_size, stride, pad, to_matrix)(x)


class Col2im(Function):
    def __init__(self, input_shape, kernel_size, stride, pad, to_matrix):
        super().__init__()
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.to_matrix = to_matrix

    def forward(self, x):
        y = col2im_array(
            x, self.input_shape, self.kernel_size, self.stride, self.pad, self.to_matrix
        )
        return y

    def backward(self, gy):
        gx = im2col_array(gy, self.kernel_size, self.stride, self.pad, self.to_matrix)
        return gx


def col2im(x, input_shape, kernel_size, stride=1, pad=0, to_matrix=True):
    return Col2im(input_shape, kernel_size, stride, pad, to_matrix)(x)


def conv2d_simple(x, Weight, b=None, stride=1, pad=0):
    x, Weight = as_variable(x), as_variable(Weight)

    # 행렬 크기 맞추는데...C가 두 군데 있네...덮어써지는데...
    N, C, H, W = x.shape
    OC, C, KH, KW = Weight.shape
    # 아무튼 출력크기 계산하고...
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    # 아래 reshape, transpose, linear 다 dezero에서 역전파 걸어놓은 함수들이라 자동 역전파 되고...
    # 입력은 im2col 트릭이고,
    col = im2col(x, (KH, KW), stride, pad, to_matrix=True)
    # 가중치는 채널, 커널크기 묶고 나머지는 출력 채널...
    # 그럼 [N, (C*KH/W)] x [(C*KH/W), OC] -> (N, OC) 행렬로 출력...
    Weight = Weight.reshape(OC, -1).transpose()
    # 여기가 실제 convolution이 되는 거고...
    t = linear(col, Weight, b)
    # 그걸 다시 분해해서 배치 x 출력 채널 x 출력 크기(H/W)로 반환...
    y = t.reshape(N, OH, OW, OC).transpose(0, 3, 1, 2)
    return y


class Conv2d(Function):
    def __init__(self, stride=1, pad=0):
        super().__init__()
        self.stride = pair(stride)
        self.pad = pair(pad)

    # 이게 convolution...
    def forward(self, x, Weight, b=None):
        xp = cuda.get_array_module(x)

        # simple 버전과 달리 가중치는 그대로, im2col은 행렬 형태가 아니라 텐서 형태 그대로...
        KH, KW = Weight.shape[2:]
        col = im2col_array(x, (KH, KW), self.stride, self.pad, to_matrix=False)

        # simple 버전(linear)과 달리 tensordot 사용...
        y = xp.tensordot(col, Weight, ((1, 2, 3), (1, 2, 3)))
        if b is not None:
            y += b
        # 이게 y = np.transpose(y, (0,3,1,2))랑 같다는 건가? 3 축을 1 자리로?
        y = xp.rollaxis(y, 3, 1)
        return y

    # 이게 convolution의 역전파이면...이게 되돌리는거 아닌가?
    def backward(self, gy):
        x, W, b = self.inputs
        gx = deconv2d(
            gy,
            W,
            b=None,
            stride=self.stride,
            pad=self.pad,
            outsize=(x.shape[2], x.shape[3]),
        )
        gW = Conv2DGradW(self)(x, gy)
        gb = None
        if b.data is not None:
            gb = gy.sum(axis=(0, 2, 3))
        return gx, gW, gb


def conv2d(x, Weight, b=None, stride=1, pad=0):
    return Conv2d(stride, pad)(x, Weight, b)


class Deconv2d(Function):
    def __init__(self, stride=1, pad=0, outsize=None):
        super().__init__()
        self.stride = pair(stride)
        self.pad = pair(pad)
        self.outsize = outsize

    def forward(self, x, Weight, b):
        xp = cuda.get_array_module(x)

        SH, SW = self.stride
        PH, PW = self.pad
        C, OC, KH, KW = Weight.shape
        N, C, H, W = x.shape
        if self.outsize is None:
            OH = get_deconv_outsize(H, KH, SH, PH)
            OW = get_deconv_outsize(W, KW, SW, PW)
        else:
            OH, OW = pair(self.outsize)
        img_shape = (N, OC, OH, OW)

        # 0, 1 축에 대해 합산하는 텐서곱? 근데 이거 convolution의 역전파일텐데 가중치는 왜 곱하지?
        gcol = xp.tensordot(Weight, x, (0, 1))
        # 축 3을 0 자리로...
        gcol = xp.rollaxis(gcol, 3)
        # 암튼 convolution 되돌리고 b는 더하고...
        img = col2im_array(
            gcol, img_shape, (KH, KW), self.stride, self.pad, to_matrix=False
        )
        # b는 채널당 하나...
        if b is not None:
            self.no_bias = True
            img += b.reshape((1, b.size, 1, 1))
        return img

    def backward(self, gy):
        x, Weight, b = self.inputs
        # convolution 되돌림의 되돌림인가? 뭔지 모르겠음...
        gx = conv2d(gy, Weight, b=None, stride=self.stride, pad=self.pad)
        # 가중치 미분은 여기서만 구해?
        f = Conv2DGradW(self)
        gW = f(gy, x)
        gb = None
        if b is not None:
            gb = gy.sum(axis=(0, 2, 3))
        return gx, gW, gb


def deconv2d(x, Weight, b=None, stride=1, pad=0, outsize=None):
    return Deconv2d(stride, pad, outsize)(x, Weight, b)


class Conv2DGradW(Function):
    def __init__(self, conv2d):
        W = conv2d.inputs[1]
        kh, kw = W.shape[2:]
        self.kernel_size = (kh, kw)
        self.stride = conv2d.stride
        self.pad = conv2d.pad

    def forward(self, x, gy):
        xp = cuda.get_array_module(x)
        col = im2col(x, self.kernel_size, self.stride, self.pad, to_matrix=False)
        gW = xp.tensordot(col, gy, ((0, 2, 3), (1, 4, 5)))
        return gW

    def backward(self, ggW):
        x, gy = self.inputs
        (gW,) = self.outputs

        xh, xw = x.shape[2:]
        gx = deconv2d(gy, gW, stride=self.stride, pad=self.pad, outsize=(xh, xw))
        ggy = conv2d(x, gW, stride=self.stride, pad=self.pad)
        return gx, ggy


# 암튼 이 아래는 풀링
class Pooling(Function):
    def __init__(self, kernel_size, stride=1, pad=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        col = im2col_array(x, self.kernel_size, self.stride, self.pad, to_matrix=False)

        N, C, KH, KW, OH, OW = col.shape
        col = col.reshape(N, C, KH * KW, OH, OW)
        self.indexes = col.argmax(axis=2)
        y = col.max(axis=2)
        return y

    def backward(self, gy):
        return Pooling2DGrad(self)(gy)


class Pooling2DGrad(Function):
    def __init__(self, mpool2d):
        self.mpool2d = mpool2d
        self.kernel_size = mpool2d.kernel_size
        self.stride = mpool2d.stride
        self.pad = mpool2d.pad
        self.input_shape = mpool2d.inputs[0].shape
        self.dtype = mpool2d.inputs[0].dtype
        self.indexes = mpool2d.indexes

    def forward(self, gy):
        xp = cuda.get_array_module(gy)

        N, C, OH, OW = gy.shape
        N, C, H, W = self.input_shape
        KH, KW = pair(self.kernel_size)

        gcol = xp.zeros((N * C * OH * OW * KH * KW), dtype=self.dtype)

        indexes = self.indexes.ravel() + xp.arange(
            0, self.indexes.size * KH * KW, KH * KW
        )

        gcol[indexes] = gy.ravel()
        gcol = gcol.reshape(N, C, OH, OW, KH, KW)
        gcol = xp.swapaxes(gcol, 2, 4)
        gcol = xp.swapaxes(gcol, 3, 5)

        gx = col2im_array(
            gcol, (N, C, H, W), self.kernel_size, self.stride, self.pad, to_matrix=False
        )
        return gx

    def backward(self, ggx):
        f = Pooling2DWithIndexes(self.mpool2d)
        return f(ggx)


class Pooling2DWithIndexes(Function):
    def __init__(self, mpool2d):
        self.kernel_size = mpool2d.kernel_size
        self.stride = mpool2d.stride
        self.pad = mpool2d.pad
        self.input_shpae = mpool2d.inputs[0].shape
        self.dtype = mpool2d.inputs[0].dtype
        self.indexes = mpool2d.indexes

    def forward(self, x):
        col = im2col_array(x, self.kernel_size, self.stride, self.pad, to_matrix=False)
        N, C, KH, KW, OH, OW = col.shape
        col = col.reshape(N, C, KH * KW, OH, OW)
        col = col.transpose(0, 1, 3, 4, 2).reshape(-1, KH * KW)
        indexes = self.indexes.ravel()
        col = col[np.arange(len(indexes)), indexes]
        return col.reshape(N, C, OH, OW)


def pooling(x, kernel_size, stride=1, pad=0):
    return Pooling(kernel_size, stride, pad)(x)


class AveragePooling(Function):
    def __init__(self, kernel_size, stride=1, pad=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.input_shape = None

    def forward(self, x):
        self.input_shape = x.shape
        col = im2col_array(x, self.kernel_size, self.stride, self.pad, to_matrix=False)
        y = col.mean(axis=(2, 3))
        return y

    def backward(self, gy):
        # TODO(Koki): This is simple implementation
        N, C, OH, OW = gy.shape
        KW, KH = pair(self.kernel_size)
        gy /= KW * KH
        gcol = broadcast_to(gy.reshape(-1), (KH, KW, N * C * OH * OW))
        gcol = gcol.reshape(KH, KW, N, C, OH, OW).transpose(2, 3, 0, 1, 4, 5)
        gx = col2im(
            gcol,
            self.input_shape,
            self.kernel_size,
            self.stride,
            self.pad,
            to_matrix=False,
        )
        return gx


def average_pooling(x, kernel_size, stride=1, pad=0):
    return AveragePooling(kernel_size, stride, pad)(x)
