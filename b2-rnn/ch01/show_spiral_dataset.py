import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dataset import spiral
import matplotlib.pyplot as plt

x, t = spiral.load_data()
print("x", x.shape)
print("t", t.shape)

N = 100
CLS_NUM = 3
markers = ["o", "x", "^"]
for i in range(CLS_NUM):
    plt.scatter(
        # 0~99, 100~199, 200~299로 나눠서, xy 좌표 뿌리면서 마커도 바꾸기..
        x[i * N : (i + 1) * N, 0],
        x[i * N : (i + 1) * N, 1],
        s=40,
        marker=markers[i],
    )
plt.show()
