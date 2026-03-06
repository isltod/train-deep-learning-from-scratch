from common.np import *  # import numpy as
from common.config import GPU


class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx):
        (W,) = self.params
        self.idx = idx
        out = W[idx]
        return out

    def backward(self, dout):
        (dW,) = self.grads
        dW[...] = 0
        # 뭔가 예전(8이하)에는 add.at을 호출해 scatter_add를 연결해서 처리했는데, 이게 없어졌다고...
        # 지금은 cupyx에 scatter_add 함수를 사용한다고...
        if GPU:
            import cupyx

            cupyx.scatter_add(dW, self.idx, dout)
        else:
            np.add.at(dW, self.idx, dout)
        return None
