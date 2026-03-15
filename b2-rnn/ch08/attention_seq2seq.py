import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from common.time_layers import *
from ch07.seq2seq import Seq2seq, Encoder
from ch07.peeky_seq2seq import PeekySeq2seq
from ch08.attention_layer import TimeAttention


class AttentionEncoder(Encoder):
    def forward(self, xs):
        xs = self.embed.forward(xs)
        hs = self.lstm.forward(xs)
        # 원래 Encoder에서는 return hs[:, -1, :]으로 TimeLSTM의 시간 차원의 마지막 벡터만 반환했는데,
        # 어텐션에서는 모든 은닉 벡터를 반환하는 것이 차이점...나머진 같다...
        return hs

    def backward(self, dhs):
        dout = self.lstm.backward(dhs)
        dout = self.embed.backward(dout)
        return dout


class AttentionDecoder:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        # 임베딩 가중치는 어휘 수 V x 표현 차원 V
        embed_W = (rn(V, D) / 100).astype("f")
        # LSTM 입력 가중치는 표현 차원 D x 히든 H(fgio 4배)
        lstm_Wx = (rn(D, 4 * H) / np.sqrt(2 * H)).astype("f")
        # LSTM 은닉은 은닉 받아서 은닉으로 보내므로, 가중치는 히든 H x 히든 H(fgio 4배)
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype("f")
        lstm_b = np.zeros(4 * H).astype("f")
        # 어텐션의 Affine은 LSTM에서 온 H + 어텐션에서 온 H 이어붙이기, 출력은 어휘 수로...Xavier 초기화
        affine_W = (rn(2 * H, V) / np.sqrt(2 * H)).astype("f")
        affine_b = np.zeros(V).astype("f")

        # 순서대로 임베딩, LSTM, 어텐션 LSTM, Affine - 이 후에 SoftmaxWithLoss로 연결
        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.attention = TimeAttention()
        self.affine = TimeAffine(affine_W, affine_b)
        layers = [self.embed, self.lstm, self.attention, self.affine]

        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs, enc_hs):
        # 인코더에서 넘어온 hs 중 마지막 h는 디코더 첫 LSTM에 상태값으로 전달
        h = enc_hs[:, -1]
        self.lstm.set_state(h)

        # 순서대로 임베딩 - LSTM - 어텐션 LSTM 거치고...
        out = self.embed.forward(xs)
        dec_hs = self.lstm.forward(out)
        c = self.attention.forward(enc_hs, dec_hs)
        # 그 맥락(단어별 유사도) 벡터와 LSTM에서 분기한 hs 이어붙이고...
        out = np.concatenate((c, dec_hs), axis=2)
        # 마지막 Affine
        score = self.affine.forward(out)

        return score

    def backward(self, dscore):
        # 마지막 Affine 층 역전파 계산하고,
        dout = self.affine.backward(dscore)

        # 이어 붙였던 단어별 유사도와 LSTM hs 나누고
        N, T, H2 = dout.shape
        H = H2 // 2
        dc, ddec_hs0 = dout[:, :, :H], dout[:, :, H:]

        # 맥락(단어별 유사도) 벡터 부분은 어텐션으로 역전파
        denc_hs, ddec_hs1 = self.attention.backward(dc)
        # LSTM hs 방향과 인코더 hs 방향 두 가지 받아서 LSTM 방향은 분기에 대한 합...
        ddec_hs = ddec_hs0 + ddec_hs1
        # LSTM 역전파 - 임베딩 역전파로 보내고...
        dout = self.lstm.backward(ddec_hs)
        # LSTM 역전파 미분값은 다시 임베딩 역전파로...
        self.embed.backward(dout)

        # 첫 번째 LSTM에 들어간 인코더에서 온 hs의 마지막 벡터에 대한 미분 받아서
        dh = self.lstm.dh
        # 인코더 방향 미분 마지막에 합(순전파에서 분기했으므로)
        denc_hs[:, -1] += dh
        # 인코더로 역전파 미분 전달
        return denc_hs

    def generate(self, enc_hs, start_id, sample_size):
        sampled = []
        sample_id = start_id
        # 인코더에서 넘어온 hs 중 마지막 h는 디코더 첫 LSTM에 상태값으로 전달
        h = enc_hs[:, -1]
        self.lstm.set_state(h)

        # 정답 문장 단어 수 또는 문자 수만큼 반복해서...
        for _ in range(sample_size):
            # 입력은 배치 1로 고려해서 변형
            x = np.array([sample_id]).reshape((1, 1))

            # 순서대로 순전파해서 점수 얻고
            out = self.embed.forward(x)
            dec_hs = self.lstm.forward(out)
            c = self.attention.forward(enc_hs, dec_hs)
            out = np.concatenate((c, dec_hs), axis=2)
            score = self.affine.forward(out)

            # 정답 단어 또는 문자는 최고 점수 하나 선택
            sample_id = np.argmax(score.flatten())
            sampled.append(sample_id)

        return sampled


class AttentionSeq2seq(Seq2seq):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        args = vocab_size, wordvec_size, hidden_size
        self.encoder = AttentionEncoder(*args)
        self.decoder = AttentionDecoder(*args)
        self.softmax = TimeSoftmaxWithLoss()

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads
