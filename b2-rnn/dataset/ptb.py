import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import urllib.request
import pickle
import numpy as np

# 다운로드 url과 파일 이름 등 상수
url_base = "https://raw.githubusercontent.com/tomsercu/lstm/master/data/"
key_file = {"train": "ptb.train.txt", "valid": "ptb.valid.txt", "test": "ptb.test.txt"}
save_file = {"train": "ptb.train.npy", "valid": "ptb.valid.npy", "test": "ptb.test.npy"}
vocab_file = "ptb.vocab.pkl"

# 소스 파일 있는 경로를 데이터 경로로
dataset_dir = os.path.dirname(os.path.abspath(__file__))


def _download(file_name):
    # 파일 이름 받아서 데이터 경로와 붙여서 path 만들고
    file_path = dataset_dir + "/" + file_name
    # 이미 있으면 받았던 파일이므로 그냥 종료
    if os.path.exists(file_path):
        return

    print("Downloading " + file_name + " ... ")

    try:
        # 그냥 받아보고(url 경로, 로컬 파일 path)
        urllib.request.urlretrieve(url_base + file_name, file_path)
    except urllib.error.URLError:
        # 안되면 ssl 인증서 검증을 비활성화해서 다시 받기
        import ssl

        ssl._create_default_https_context = ssl._create_unverified_context
        urllib.request.urlretrieve(url_base + file_name, file_path)

    print("Done")


def load_vocab():
    # 어휘 파일 경로 설정하고
    vocab_path = dataset_dir + "/" + vocab_file

    # 이미 있으면 받았던 파일이므로 읽어들여서 반환
    if os.path.exists(vocab_path):
        with open(vocab_path, "rb") as f:
            word_to_id, id_to_word = pickle.load(f)
        return word_to_id, id_to_word

    # 없으면 새로 받아야 하는데...
    word_to_id = {}
    id_to_word = {}
    # 왜 없으면 train을 받는 거지?
    data_type = "train"
    file_name = key_file[data_type]
    file_path = dataset_dir + "/" + file_name

    _download(file_name)

    # 줄바꿈 대신 <eos>를 넣고, 모든 줄이 연결된 하나의 덩어리로 처리
    # 좌우 여백 없애고, 빈칸으로 끊어서 리스트로...
    words = open(file_path).read().replace("\n", "<eos>").strip().split()

    # 단어 사전 만들고(id2word, word2id)
    for i, word in enumerate(words):
        if word not in word_to_id:
            tmp_id = len(word_to_id)
            word_to_id[word] = tmp_id
            id_to_word[tmp_id] = word

    # 피클로 저장하고, 단어사전 반환
    with open(vocab_path, "wb") as f:
        pickle.dump((word_to_id, id_to_word), f)

    return word_to_id, id_to_word


def load_data(data_type):
    # 이건 뭐..왜 valid만 val로 해도 되게 하는 거냐...
    if data_type == "val":
        data_type = "valid"
    # 파일 path 만들고
    save_path = dataset_dir + "/" + save_file[data_type]

    # 단어 사전 받고
    word_to_id, id_to_word = load_vocab()

    # 이미 있다면...아마 위 load_vocab에서도 있는 파일 읽어들였겠지만, 그 파일 읽어서 말뭉치 반환...
    if os.path.exists(save_path):
        # npy, npz, pickle 파일 불러오기
        corpus = np.load(save_path)
        return corpus, word_to_id, id_to_word

    # 없다면, 새로 받고? 없다면 위에서 load_vocab에서 이미 받았을텐데...
    file_name = key_file[data_type]
    file_path = dataset_dir + "/" + file_name
    _download(file_name)

    # 여기선 load_vocab에서 받은 word2id 사전으로 말뭉치만 만들어서...
    words = open(file_path).read().replace("\n", "<eos>").strip().split()
    corpus = np.array([word_to_id[w] for w in words])

    # 저장하고 반환
    np.save(save_path, corpus)
    return corpus, word_to_id, id_to_word


if __name__ == "__main__":
    for data_type in ("train", "val", "test"):
        load_data(data_type)
