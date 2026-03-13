import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from common.time_layers import *
from common.np import *
from common.base_model import BaseModel


class BetterRnnlm(BaseModel):
    def __init__(
        self, vocab_size=10000, wordvec_size=650, hidden_size=650, dropout_ratio=0.5
    ):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        # 가중치 초기화
        embed_W = (rn(V, D) / 100).astype("f")
        # LSTM이 두 층으로 늘어나는데...
        lstm_Wx1 = (rn(D, 4 * H) / np.sqrt(D)).astype("f")
        lstm_Wh1 = (rn(H, 4 * H) / np.sqrt(H)).astype("f")
        lstm_b1 = np.zeros(4 * H).astype("f")
        # 두 번째 LSTM도 히든이니까...여긴 다 Hx4H
        lstm_Wx2 = (rn(H, 4 * H) / np.sqrt(H)).astype("f")
        lstm_Wh2 = (rn(H, 4 * H) / np.sqrt(H)).astype("f")
        lstm_b2 = np.zeros(4 * H).astype("f")
        # Affine은 Embedding과 가중치 공유...편향만 따로 만드네...
        affine_b = np.zeros(V).astype("f")

        # 계층 생성 (드롭아웃, 2층 LSTM, 가중치 공유)
        self.layers = [
            TimeEmbedding(embed_W),
            TimeDropout(dropout_ratio),
            TimeLSTM(lstm_Wx1, lstm_Wh1, lstm_b1, stateful=True),
            TimeDropout(dropout_ratio),
            TimeLSTM(lstm_Wx2, lstm_Wh2, lstm_b2, stateful=True),
            TimeDropout(dropout_ratio),
            TimeAffine(embed_W.T, affine_b),  # 가중치 공유 (Weight Tying)
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.lstm_layers = [self.layers[2], self.layers[4]]
        self.drop_layers = [self.layers[1], self.layers[3], self.layers[5]]

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    # forward -> predict 메서드 체인은 훈련에만 드롭아웃을 적용하므로, 인수로 train_flg를 가진다..
    def predict(self, xs, train_flg=False):
        for layer in self.drop_layers:
            layer.train_flg = train_flg

        for layer in self.layers:
            xs = layer.forward(xs)
        return xs

    def forward(self, xs, ts, train_flg=True):
        score = self.predict(xs, train_flg)
        loss = self.loss_layer.forward(score, ts)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def reset_state(self):
        for layer in self.lstm_layers:
            layer.reset_state()
