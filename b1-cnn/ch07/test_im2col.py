import numpy as np
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from common.util import im2col

# 세로 가로 7x7, RGB 이미지 1장
x1 = np.random.rand(1, 3, 7, 7)
col1 = im2col(x1, 5, 5, stride=1, pad=0)
print(col1.shape)

x2 = np.random.rand(10, 3, 7, 7)
col2 = im2col(x2, 5, 5, stride=1, pad=0)
print(col2.shape)
