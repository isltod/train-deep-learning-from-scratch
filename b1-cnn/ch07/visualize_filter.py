import numpy as np
import matplotlib.pyplot as plt
from simple_convnet import SimpleConvNet


# filters는 가중치, 그림 nx는 가로 배치 수
def filter_show(filters, nx=8, margin=3, scale=10):
    FN, C, FH, FW = filters.shape
    # 필터 수를 가로 배치 수로 나누면 세로 배치 수
    ny = int(np.ceil(FN / nx))

    fig = plt.figure()
    # 그림 상하좌우 여백, 그림 사이 여백
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(FN):
        ax = fig.add_subplot(ny, nx, i + 1, xticks=[], yticks=[])
        ax.imshow(filters[i, 0], cmap=plt.cm.gray_r, interpolation="nearest")
    plt.show()


network = SimpleConvNet()
# 필터 시각화
filter_show(network.params["W1"])

network.load_params("params.pkl")
filter_show(network.params["W1"])
