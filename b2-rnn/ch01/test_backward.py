import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from common.np import *  # import numpy as

D, N = 8, 7

# repeat 노드
x = np.random.randn(1, D)  # (1, 8)
# 순전파에서 8차원 벡터를 7개 노드로 보낸다면
y = np.repeat(x, N, axis=0)  # (7, 8)
dy = np.random.randn(N, D)  # (7, 8)
# 역전파에서는 7개 노드의 값을 더해서 8차원 벡터로 만든다...
# keedims=True를 안 쓰면 (8,) 모양이 되서 열벡터가 된다...(1,8)은 행벡터...
# dx = np.sum(dy, axis=0)  # (1, 8)
dx = np.sum(dy, axis=0, keepdims=True)  # (1, 8)
print(dx)

# sum 노드
x = np.random.randn(N, D)  # (7, 8)
# 이건 sum과 repeat 위치가 repeat 노드와 정반대...
# 여기도 keepdims 없으면 (8,) 돼서 열벡터
y = np.sum(x, axis=0, keepdims=True)  # (1, 8)
dy = np.random.randn(D)  # (1, 8)
# sum 노드는 분기해서 보내므로 같은 값을 7번 반복해서 배치 차원으로 나열...
dx = np.repeat(dy, N)  # (7, 8)
print(dx)
