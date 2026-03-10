import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from common.util import most_similar, analogy
import pickle

# CBOW 모델 학습 매개변수 불러들이고
pkl_file = os.path.join(os.path.dirname(__file__), "cbow_params.pkl")

with open(pkl_file, "rb") as f:
    params = pickle.load(f)
    # 내용은 단어벡터, 단어->ID 사전, ID->단어 사전 세 가지...
    word_vecs = params["word_vecs"]
    word_to_id = params["word_to_id"]
    id_to_word = params["id_to_word"]

# 비슷한 단어 5개 뽑기
querys = ["you", "year", "car", "toyota"]
for query in querys:
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)

# 단어 관계 유추 - 이게 생각보다 안 좋은데?
print("-" * 50)
analogy("king", "man", "queen", word_to_id, id_to_word, word_vecs)
analogy("take", "took", "go", word_to_id, id_to_word, word_vecs)
analogy("car", "cars", "child", word_to_id, id_to_word, word_vecs)
analogy("good", "better", "bad", word_to_id, id_to_word, word_vecs)
