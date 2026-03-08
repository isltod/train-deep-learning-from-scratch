import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from common.util import preprocess, create_co_matrix, ppmi
import matplotlib.pyplot as plt

text = "You say goodbye and I say hello."
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
W = ppmi(C)

# SVD
U, S, V = np.linalg.svd(W)

np.set_printoptions(precision=3)
print(C[0])
print(W[0])
print(U[0])

# 7차원 벡터로 표현된 단어를 앞의 2개 차원만 선택해서 차원축소...왜 2만?
for word, word_id in word_to_id.items():
    # word를 U 행렬의 [id, 0] 값과 [id, 1] 값 좌표에 표시
    plt.annotate(word, (U[word_id, 0], U[word_id, 1]))

plt.scatter(U[:, 0], U[:, 1], alpha=0.5)
plt.show()
