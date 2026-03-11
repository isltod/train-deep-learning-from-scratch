import numpy as np
import matplotlib.pyplot as plt

N = 2  # 배치
H = 3  # 은닉 상태 벡터 차원
T = 20  # RNN 수

dh = np.ones((N, H))
np.random.seed(3)
# Wh = np.random.randn(H, H)
Wh = np.random.randn(H, H) * 0.5

norm_list = []
for t in range(T):
    dh = np.dot(dh, Wh.T)
    norm = np.sqrt(np.sum(dh**2)) / N
    norm_list.append(norm)

print(norm_list)

plt.plot(np.arange(len(norm_list)), norm_list)
plt.xticks([0, 4, 9, 14, 19], [1, 5, 10, 15, 20])
plt.xlabel("시간 크기")
plt.ylabel("노름")
plt.show()
