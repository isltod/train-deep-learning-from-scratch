import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from common.util import create_co_matrix, cos_similarity, preprocess

text = "You say goodbye and I say hello."
# text를 단어 ID로 바꾼 리스트, 단어별 ID, ID별 단어 사전 받고
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
# 동시발생 행렬 만들고
C = create_co_matrix(corpus, vocab_size)

# 단어를 ID로 바꿔서 동시발생 행렬에서 해당 인덱스를 찾으면 그걸 그 단어의 벡터로...
c0 = C[word_to_id["you"]]
c1 = C[word_to_id["i"]]
# 내적 기반 상관계수
print(cos_similarity(c0, c1))
