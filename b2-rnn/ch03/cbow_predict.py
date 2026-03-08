import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from common.layers import MatMul

# 단어의 원핫벡터, 행은 배치(현재 1), 열은 단어(현재 7 차원)
c0 = np.array([[1, 0, 0, 0, 0, 0, 0]])
c1 = np.array([[0, 0, 1, 0, 0, 0, 0]])

# 가중치인데...은닉층이 (1, 3)으로 줄였다 늘려야 하고...
W_in = np.random.randn(7, 3)
W_out = np.random.randn(3, 7)

# 맥락(윈도우) 단어는 2개가 입력되므로, 입력 층은 (7, 3) 두 개...
in_layer0 = MatMul(W_in)
in_layer1 = MatMul(W_in)
# 출력 층은 다시 (1, 7) 복원해야 하니까 (3, 7)
out_layer = MatMul(W_out)

# 두 단어의 임시 결과 받아서 평균내고
h0 = in_layer0.forward(c0)
h1 = in_layer1.forward(c1)
h = 0.5 * (h0 + h1)
# 그 결과를 출력으로...이걸 소프트맥스 해서 원핫으로 바꿔야 단어가 되겠지...
s = out_layer.forward(h)

print(s)
