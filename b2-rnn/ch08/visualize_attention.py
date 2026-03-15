import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib.pyplot as plt
from dataset import sequence
from attention_seq2seq import AttentionSeq2seq

(x_train, t_train), (x_test, t_test) = sequence.load_data("date.txt")
char_to_id, id_to_char = sequence.get_vocab()

x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]

vocab_size = len(char_to_id)
wordvec_size = 16
hidden_size = 256

model = AttentionSeq2seq(vocab_size, wordvec_size, hidden_size)
model.load_params()

_idx = 0


def visualize(attention_map, row_labels, col_labels):
    # 한 화면(Figure)에 여러 개의 그래프(Axes)를 효율적으로 생성하고 제어하는 함수
    fig, ax = plt.subplots()
    # 2D 데이터 배열을 기반으로 사각형 격자에서 각 값을 색으로 표현
    ax.pcolor(attention_map, cmap=plt.cm.Greys_r, vmin=0.0, vmax=1.0)

    # 특정 Axes(그래프 영역)의 배경색을 설정하는 메서드
    ax.patch.set_facecolor("black")
    ax.set_yticks(np.arange(attention_map.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(attention_map.shape[1]) + 0.5, minor=False)
    # y축 방향 뒤집기, 0~100 -> 100~0
    ax.invert_yaxis()
    ax.set_xticklabels(row_labels, minor=False)
    ax.set_yticklabels(col_labels, minor=False)

    global _idx
    _idx += 1
    plt.show()


np.random.seed(1984)
for _ in range(5):
    idx = [np.random.randint(0, len(x_test))]
    x = x_test[idx]
    t = t_test[idx]

    model.forward(x, t)
    d = model.decoder.attention.attention_weights
    d = np.array(d)
    attention_map = d.reshape(d.shape[0], d.shape[2])

    # 출력하기 위해서 반전시킨다고? 왜?
    attention_map = attention_map[:, ::-1]
    x = x[:, ::-1]

    row_labels = [id_to_char[c] for c in x[0]]
    col_labels = [id_to_char[c] for c in t[0]]
    col_labels = col_labels[1:]

    visualize(attention_map, row_labels, col_labels)
