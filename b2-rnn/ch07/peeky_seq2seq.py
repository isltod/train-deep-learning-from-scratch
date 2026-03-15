import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from common.time_layers import *
from ch07.seq2seq import Seq2seq, Encoder


class PeekyDecoder:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        embed_W = (rn(V, D) / 100).astype("f")
        # Peeky에서는 LSTM의 입력으로 (h + embed)가 들어가므로 입력 차원이 H + D
        lstm_Wx = (rn(H + D, 4 * H) / np.sqrt(H + D)).astype("f")
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype("f")
        lstm_b = np.zeros(4 * H).astype("f")
        # Affine의 입력으로도 (h + lstm_out)이 들어가므로 입력 차원이 H + H
        affine_W = (rn(H + H, V) / np.sqrt(H + H)).astype("f")
        affine_b = np.zeros(V).astype("f")

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.affine = TimeAffine(affine_W, affine_b)

        self.params, self.grads = [], []
        for layer in (self.embed, self.lstm, self.affine):
            self.params += layer.params
            self.grads += layer.grads
        self.cache = None

    def forward(self, xs, h):
        # 원래 X는 배치 N x T 칸(T개의 문자 ID)
        N, T = xs.shape
        # 원래 h는 배치 N x 히든 크기 H인데...
        N, H = h.shape

        self.lstm.set_state(h)

        # 배치 N에 대해, 단어(문자) ID T개를 가지고, W에서 T행의 표현 D 차원을 꺼내서 out 만들기...
        out = self.embed.forward(xs)
        # 324쪽 그림 7-26 참고, h를 문자 수 T만큼 복제해서(NTH) 모든 LSTM에 입력, axis=0은 행 방향 T 반복
        hs = np.repeat(h, T, axis=0).reshape(N, T, H)
        # Peeky 방식에서는 LSTM 출력과 Encoder의 h를 단순 이어붙인다...
        # 배치 x 단어(문자) 수 x (단어(문자) 표현 차원 + 히든 표현 차원)
        out = np.concatenate((hs, out), axis=2)

        out = self.lstm.forward(out)
        # Affine 계층에도 h를 추가 - 여기는 배치 x 단어(문자) 수 x (인코더 히든 차원 + 여기 히든 차원)
        out = np.concatenate((hs, out), axis=2)

        score = self.affine.forward(out)
        self.cache = H
        return score

    def backward(self, dscore):
        H = self.cache

        dout = self.affine.backward(dscore)
        # 순전파 Affine 직전에 이어 붙였던 인코더 히든 차원(H)과 디코더 히든 차원(H)을 분리
        dout, dhs0 = dout[:, :, H:], dout[:, :, :H]
        # 그 중 LSTM 관련 부분만 역전파
        dout = self.lstm.backward(dout)
        # 순전파 LSTM 직전에 이어붙였던 인코더 히든 차원(H)과 단어(문자) 표현 차원(D) 분리
        dembed, dhs1 = dout[:, :, H:], dout[:, :, :H]
        # 그 중 Embedding 관련 부분만 역전파
        self.embed.backward(dembed)

        # h 부분은 따로 합하기만...분기해서 들어갔으니 역으로 합한다?
        dhs = dhs0 + dhs1
        # 그리고 T만큼 np.repeat 했던 것 다시 합치고, LSTM에 state로 넣은 것 합치기?
        dh = self.lstm.dh + np.sum(dhs, axis=1)
        return dh

    def generate(self, h, start_id, sample_size):
        sampled = []
        char_id = start_id
        self.lstm.set_state(h)

        H = h.shape[1]
        peeky_h = h.reshape(1, 1, H)
        for _ in range(sample_size):
            x = np.array([char_id]).reshape((1, 1))
            out = self.embed.forward(x)

            out = np.concatenate((peeky_h, out), axis=2)
            out = self.lstm.forward(out)
            out = np.concatenate((peeky_h, out), axis=2)
            score = self.affine.forward(out)

            char_id = np.argmax(score.flatten())
            sampled.append(char_id)

        return sampled


class PeekySeq2seq(Seq2seq):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        self.encoder = Encoder(V, D, H)
        self.decoder = PeekyDecoder(V, D, H)
        self.softmax = TimeSoftmaxWithLoss()

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads
