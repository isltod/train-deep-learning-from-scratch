from dezero import Layer
from dezero import utils
import dezero.functions as F
import dezero.layers as L


class Model(Layer):
    # plot_dot_graph를 이용하므로 tmp 아래에 model.png 파일 만든다...
    # 근데 이건 편의상 넣은거지 전혀 필요없는데...그럼 사실 Layer 클래스로 다 된건데...
    def plot(self, *inputs, to_file="model.png"):
        y = self.forward(*inputs)
        return utils.plot_dot_graph(y, verbose=True, to_file=to_file)


# Multi-Layer Perceptron
class MLP(Model):
    # fc - Fully Connected
    def __init__(self, fc_output_sizes, activation=F.sigmoid):
        super().__init__()
        self.activation = activation
        self.layers = []

        for i, out_size in enumerate(fc_output_sizes):
            # fc_output_sizes로 지정된 만큼 Linear 레이어 만들고 순서대로 이름줘서 저장...
            layer = L.Linear(out_size)
            # self.l1, self.l2...이런 식으로 코딩할 수 없으니 setattr 이용...
            setattr(self, "l" + str(i), layer)
            # 그리고 인스턴스 변수에도 또 저장...이건 forward 반복문 돌릴려고..
            self.layers.append(layer)

    def forward(self, x):
        # 마지막을 제외한 모든 레이어에 forward + 활성화함수 적용
        for l in self.layers[:-1]:
            x = self.activation(l(x))
        # 마지막은 그냥 forward로 반환
        return self.layers[-1](x)
