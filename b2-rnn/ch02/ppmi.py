import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from common.util import preprocess, create_co_matrix, ppmi

text = "You say goodbye and I say hello."
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
W = ppmi(C)

# 유효 자릿수를 소숫점 아래 3 자리로 표시
np.set_printoptions(precision=3)
print("동시발생 행렬")
print(C)
print("-" * 50)
print("PPMI")
print(W)
