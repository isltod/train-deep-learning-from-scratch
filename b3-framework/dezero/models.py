from dezero import Layer
from dezero import utils
import dezero.functions as F
import dezero.layers as L
import numpy as np


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


class VGG16(Model):
    WEIGHTS_PATH = (
        "https://github.com/koki0702/dezero-models/releases/download/v0.1/vgg16.npz"
    )

    def __init__(self, pretrained=False):
        super().__init__()
        self.conv1_1 = L.Conv2d(64, kernel_size=3, stride=1, pad=1)
        self.conv1_2 = L.Conv2d(64, kernel_size=3, stride=1, pad=1)
        self.conv2_1 = L.Conv2d(128, kernel_size=3, stride=1, pad=1)
        self.conv2_2 = L.Conv2d(128, kernel_size=3, stride=1, pad=1)
        self.conv3_1 = L.Conv2d(256, kernel_size=3, stride=1, pad=1)
        self.conv3_2 = L.Conv2d(256, kernel_size=3, stride=1, pad=1)
        self.conv3_3 = L.Conv2d(256, kernel_size=3, stride=1, pad=1)
        self.conv4_1 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv4_2 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv4_3 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv5_1 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv5_2 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv5_3 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.fc6 = L.Linear(4096)
        self.fc7 = L.Linear(4096)
        self.fc8 = L.Linear(1000)

        if pretrained:
            weights_path = utils.get_file(VGG16.WEIGHTS_PATH)
            self.load_weights(weights_path)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = F.pooling(x, 2, 2)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = F.pooling(x, 2, 2)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = F.pooling(x, 2, 2)
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = F.pooling(x, 2, 2)
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = F.pooling(x, 2, 2)
        x = F.reshape(x, (x.shape[0], -1))
        x = F.dropout(F.relu(self.fc6(x)))
        x = F.dropout(F.relu(self.fc7(x)))
        x = self.fc8(x)
        return x

    @staticmethod
    def preprocess(image, size=(224, 224), dtype=np.float32):
        # 책은 BGR 순서로 재정렬한다는데, 여긴 RGB인데?
        image = image.convert("RGB")
        if size:
            image = image.resize(size)
        image = np.asarray(image, dtype=dtype)
        # 이건 뭐지? 0, 1 축은 모든 데이터, 마지막은? stride가 -1이면 거꾸로 뒤집으란 얘긴가?
        image = image[:, :, ::-1]
        # 이 숫자는 뭔데 그냥 빼냐? 이미지넷 데이터들의 채널별 평균인가?
        image -= np.array([103.939, 116.779, 123.68], dtype=dtype)
        image = image.transpose((2, 0, 1))
        return image
