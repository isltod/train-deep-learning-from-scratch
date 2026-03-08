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
