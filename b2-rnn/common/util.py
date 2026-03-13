import os
import sys
from common.np import *


# max_norm을 넘어가면 기울기를 줄인다는데...
def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad**2)
    total_norm = np.sqrt(total_norm)

    # 기울기 놈이 크면 1보다 작아지고, 그 비율로 기울기 줄이기...
    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate
    return grads


def preprocess(text):
    text = text.lower()
    text = text.replace(".", " .")
    words = text.split(" ")
    word_to_id = {}
    id_to_word = {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word
    corpus = np.array([word_to_id[w] for w in words])
    return corpus, word_to_id, id_to_word


def create_co_matrix(corpus, vocab_size, window_size=1):
    # 단어 ID 목록 - 문장을 그대로, 단어를 ID로 변경한 리스트
    corpus_size = len(corpus)
    # 어휘 수
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            # i에 따라 window_size까지 넓혀가면서
            left_idx = idx - i
            right_idx = idx + i

            # 왼쪽 단어가 있다면, 그 단어의 위치에 1 추가
            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                # 첫 번째 인덱스는 단어, 두 번째 인덱스는 동시 출현
                co_matrix[word_id, left_word_id] += 1

            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1

    return co_matrix


def cos_similarity(x, y, eps=1e-8):
    # 분모는 스칼라, 분자는 벡터, x, y 각각을 정규화해서 곱한다...상관계수
    nx = x / (np.sqrt(np.sum(x**2)) + eps)
    ny = y / (np.sqrt(np.sum(y**2)) + eps)
    return np.dot(nx, ny)


def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    # 검색어가 단어 목록에 없으면 종료
    if query not in word_to_id:
        print("%s(을)를 찾을 수 없습니다." % query)
        return

    print("\n[query] " + query)
    # 검색어를 벡터로 읽고
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    # 유사도를 넣을 배열 0으로 초기화
    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        # 단어별로 검색어와 코사인 유사도 구하고 단어 ID에 맞춰 유사도 배열에 저장
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    count = 0
    # argsort는 정렬된 인덱스 배열을 반환
    for i in (-1 * similarity).argsort():
        # 자기 자신과 상관이 가장 높을테니 그건 패스
        if id_to_word[i] == query:
            continue
        print(" %s: %s" % (id_to_word[i], similarity[i]))
        count += 1
        if count >= top:
            return


def ppmi(C, verbose=False, eps=1e-8):
    # ppmi는 각 단어별 전체 단어들에 대해서 다 구하는 모양...그래서 동시행렬과 같은 모양...
    M = np.zeros_like(C, dtype=np.float32)
    # 그냥 문장에서 몇 번 나왔냐가 아니라, 단어쌍으로 몇 번 나왔냐를 세는 모양...
    # 그럼 말뭉치라는 것이 윈도우로 본 단어 쌍을 말하는 건가?
    N = np.sum(C)
    # 행별로 합 - 단어별로 단어쌍에 몇 번 나왔나 합...
    # 근데 동시발생 행렬이 대칭행렬이라 열 방향으로 합해도 결과는 같다...
    S = np.sum(C, axis=0)
    total = C.shape[0] * C.shape[1]
    cnt = 0

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / (S[j] * S[i]) + eps)
            M[i, j] = max(0, pmi)

            if verbose:
                cnt += 1
                if cnt % (total // 100) == 0:
                    print("%.1f%% 완료" % (100 * cnt / total))
    return M


def create_contexts_target(corpus, window_size=1):
    # 양 끝 단어(윈도우 크기 고려해서)는 제외하고..순서대로 타겟
    target = corpus[window_size:-window_size]
    contexts = []

    # 여기도 윈도우 크기 고려해서 끝단 단어는 제외하고 돌면서
    for idx in range(window_size, len(corpus) - window_size):
        cs = []
        # 윈도우만큼 이전부터 다음까지를 순서대로 담고
        for t in range(-window_size, window_size + 1):
            if t == 0:
                continue
            cs.append(corpus[idx + t])
        # 이걸 다시 context에 담으면, 행은 target, 열은 window 내 context...
        contexts.append(cs)

    return np.array(contexts), np.array(target)


def convert_one_hot(corpus, vocab_size):
    N = corpus.shape[0]
    # 받은 말뭉치가 1차원이면, target이란 말이고...
    if corpus.ndim == 1:
        # 원핫은 원래 크기를 행으로 놓고, 열은 어휘 수로 초기화 - 0
        one_hot = np.zeros((N, vocab_size), dtype=np.int32)
        for idx, word_id in enumerate(corpus):
            # 인덱스 위치만 1로 바꾸면 원핫
            one_hot[idx, word_id] = 1
    # 아니고 2차원이면, contexts라는 얘기고
    elif corpus.ndim == 2:
        C = corpus.shape[1]
        # 세 번째 차원으로 어휘 수만큼 0 초기화
        one_hot = np.zeros((N, C, vocab_size), dtype=np.int32)
        # target - context 별로 돌면서
        for idx_0, word_ids in enumerate(corpus):
            # context 안의 각 단어마다 원핫 처리...
            for idx_1, word_id in enumerate(word_ids):
                one_hot[idx_0, idx_1, word_id] = 1

    return one_hot


# 배열을 cupy <-> numpy 변환
def to_cpu(x):
    import numpy

    if type(x) == numpy.ndarray:
        return x
    return np.asnumpy(x)


def to_gpu(x):
    import cupy

    if type(x) == cupy.ndarray:
        return x
    return cupy.asarray(x)


def normalize(x):
    if x.ndim == 2:
        s = np.sqrt((x * x).sum(1))
        x /= s.reshape((s.shape[0], 1))
    elif x.ndim == 1:
        s = np.sqrt((x * x).sum())
        x /= s
    return x


# a/b/c는 king/man/queen 같은 관련 단어들
def analogy(a, b, c, word_to_id, id_to_word, word_matrix, top=5, answer=None):
    # 검색어가 단어 목록에 없으면 종료
    for word in (a, b, c):
        if word not in word_to_id:
            print("%s(을)를 찾을 수 없습니다." % word)
            return

    print("\n[analogy] " + a + ":" + b + " = " + c + ":?")
    # 검색어를 벡터로 읽고
    a_vec, b_vec, c_vec = (
        word_matrix[word_to_id[a]],
        word_matrix[word_to_id[b]],
        word_matrix[word_to_id[c]],
    )
    # king - man + woman 같은 단어 유추 연산
    query_vec = b_vec - a_vec + c_vec
    # 정규화...유사도 구하는데 분모 norm이 다 1로 떨어지도록...단어 벡터들은 정규화가 되어 있나?
    # 일단 출발이 randn이라 정규분포에서 추출하긴 하는데...그래도 앞에 0.01을 곱하고 시작하는데?
    query_vec = normalize(query_vec)

    # 단어별 내적을 다 구하는데...이게 유사도...
    similarity = np.dot(word_matrix, query_vec)
    # 요구되는 답이 있다면 그 답에 대한 유사도만 보고
    if answer is not None:
        print(
            "==>"
            + answer
            + ":"
            + str(np.dot(word_matrix[word_to_id[answer]], query_vec))
        )

    # 아니면 top에 지정된 숫자만큼 순서대로 유사한 단어 보고...
    count = 0
    for i in (-1 * similarity).argsort():
        if np.isnan(similarity[i]):
            continue
        if id_to_word[i] in (a, b, c):
            continue
        print(" %s: %s" % (id_to_word[i], similarity[i]))

        count += 1
        if count >= top:
            return


def eval_perplexity(model, corpus, batch_size=10, time_size=35):
    print("퍼플렉서티 평가 중...")
    corpus_size = len(corpus)
    total_loss = 0
    # 반복 변수에서 0부터 시작하기 위해서 1 빼기...
    max_iters = (corpus_size - 1) // (batch_size * time_size)
    jump = (corpus_size - 1) // batch_size

    for iters in range(max_iters):
        # NxT
        xs = np.zeros((batch_size, time_size), dtype=np.int32)
        ts = np.zeros((batch_size, time_size), dtype=np.int32)
        # iters 반복 따라 0, 35, 70, ... 증가
        time_offset = iters * time_size
        # i * jump = [1000, 2000, ... , 10000] + time_offset (0, 35, 70, ...)
        offsets = [time_offset + (i * jump) for i in range(batch_size)]
        # 배치, time_size 고려해서 순차적으로 문제 단어와 정답 읽고
        for t in range(time_size):
            for i, offset in enumerate(offsets):
                xs[i, t] = corpus[offset + t]
                ts[i, t] = corpus[offset + t + 1]

        # 손실 계산
        try:
            loss = model.forward(xs, ts, train_flg=False)
        except TypeError:
            loss = model.forward(xs, ts)
        total_loss += loss

        sys.stdout.write("\r%d / %d" % (iters, max_iters))
        sys.stdout.flush()

    print("")
    # 최종 퍼플렉서티...230쪽 식 5.12, 13 참고
    ppl = np.exp(total_loss / max_iters)
    return ppl


def eval_seq2seq(model, question, correct, id_to_char, verbose=False, is_reverse=False):
    # 이게 정답지 배열인가? 1차원으로 만들고
    correct = correct.flatten()
    # 머릿글자
    start_id = correct[0]
    # 나머지 정답들...
    correct = correct[1:]
    # 생성된 추측들...
    guess = model.generate(question, start_id, len(correct))

    # 다 문자열로 변환
    question = "".join([id_to_char[int(c)] for c in question.flatten()])
    correct = "".join([id_to_char[int(c)] for c in correct])
    guess = "".join([id_to_char[int(c)] for c in guess])

    if verbose:
        # 이건 뭔가? 뒤에서 나오는 트릭인가?
        if is_reverse:
            # 처음부터 끝까지 -1, 반대 방향으로...뒤집기
            question = question[::-1]

        # 뭔가 화면에 맞고 틀리고를 구분해주는 트릭들...
        colors = {"ok": "\033[92m", "fail": "\033[91m", "close": "\033[0m"}
        print("Q", question)
        print("T", correct)

        is_windows = os.name == "nt"

        if correct == guess:
            mark = colors["ok"] + "☑" + colors["close"]
            if is_windows:
                mark = "O"
            print(mark + " " + guess)
        else:
            mark = colors["fail"] + "☒" + colors["close"]
            if is_windows:
                mark = "X"
            print(mark + " " + guess)
        print("---")

    return 1 if guess == correct else 0
