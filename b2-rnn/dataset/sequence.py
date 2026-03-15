import sys

sys.path.append("..")
import numpy as np
import os

id_to_char = {}
char_to_id = {}


def _update_vocab(txt):
    chars = list(txt)

    for i, char in enumerate(chars):
        if char not in char_to_id:
            tmp_id = len(char_to_id)
            char_to_id[char] = tmp_id
            id_to_char[tmp_id] = char


def load_data(file_name="addition.txt", seed=1984):
    file_path = os.path.dirname(os.path.abspath(__file__)) + "/" + file_name

    if not os.path.exists(file_path):
        print("No file: " + file_name)
        return None

    questions, answers = [], []

    for line in open(file_path, "r"):
        # 줄마다 돌면서 답 구분자 이전은 문제로, 이후는 답으로...
        idx = line.find("_")
        questions.append(line[:idx])
        answers.append(line[idx:-1])

    for i in range(len(questions)):
        _update_vocab(questions[i])
        _update_vocab(answers[i])

    # 문제는 7칸, 답은 5칸으로 맞춰놔서 아무거나 0번 골라서... x는 (50,000, 7), t는 (50,000, 5)
    x = np.zeros((len(questions), len(questions[0])), dtype=int)
    t = np.zeros((len(questions), len(answers[0])), dtype=int)

    # 문제, 답 한 줄씩, 거기 공백 등 문자를 순서대로 id 리스트로 i 번째 요소로...
    for i, sentence in enumerate(questions):
        x[i] = [char_to_id[c] for c in list(sentence)]
        ㅁㅁ = 0

    for i, sentence in enumerate(answers):
        t[i] = [char_to_id[c] for c in list(sentence)]

    # 문제와 정답 같이 섞기
    indices = np.arange(len(x))
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(indices)
    x = x[indices]
    t = t[indices]

    # 훈련 90%, 테스트 10%
    split_at = len(x) - len(x) // 10
    (x_train, x_test) = x[:split_at], x[split_at:]
    (t_train, t_test) = t[:split_at], t[split_at:]

    return (x_train, t_train), (x_test, t_test)


def get_vocab():
    return char_to_id, id_to_char
