import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from common.util import create_co_matrix, preprocess


text = "You say goodbye and I say hello."
corpus, word_to_id, id_to_word = preprocess(text)
print(corpus)
print(id_to_word)
print(word_to_id)

co_matrix = create_co_matrix(corpus, len(word_to_id), window_size=1)
print(co_matrix)
