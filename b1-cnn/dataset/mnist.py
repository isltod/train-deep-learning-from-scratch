import gzip
import numpy as np
import pickle
import os, os.path
import urllib.request

url_base = "http://yann.lecun.com/exdb/mnist/"
data_gz_files = {
    "train_img": "train-images-idx3-ubyte.gz",
    "train_label": "train-labels-idx1-ubyte.gz",
    "test_img": "t10k-images-idx3-ubyte.gz",
    "test_label": "t10k-labels-idx1-ubyte.gz",
}

# 현재 스크립트 파일의 절대 경로에서 디렉터리까지만 추출
dataset_dir = os.path.dirname(os.path.abspath(__file__))
saved_mnist_file = dataset_dir + "/mnist.pkl"

train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784


def _download(file_name):
    file_path = dataset_dir + "/" + file_name
    # 이미 받았으면 다운로드 안함
    if os.path.exists(file_path):
        return

    print(file_name + "을(를) 다운로드 합니다...")
    # 크롤링으로 간단하게 파일을 다운로드...
    urllib.request.urlretrieve(url_base + file_name, file_path)
    print("다운로드 완료.")


def download_mnist():
    for v in data_gz_files.values():
        _download(v)


def _load_label(file_name):
    file_path = dataset_dir + "/" + file_name
    print(file_name + "을(를) 넘파이 배열로 변환...")
    # 이게 gzip 파일 압축을 풀어서 여는 모양...
    with gzip.open(file_path, "rb") as f:
        # 원본 버퍼의 메모리를 공유해서 빨리 읽어온다...
        # f.read()가 버퍼, 부호 없는 정수형, 오프셋 헤더 바이트 수
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    # 로우 수는 데이터 전체 길이에맞춰 자동 계산하고 컬럼 수는 1
    # -> 컬럼 벡터 만들기
    print("변환 전: ", data.shape)
    data = data.reshape(-1, 1)
    print("변환 후: ", data.shape)
    print("변환 완료.")
    return data


def _load_img(file_name):
    file_path = dataset_dir + "/" + file_name
    print(file_name + "을(를) 넘파이 배열로 변환...")
    with gzip.open(file_path, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, img_size)
    print("변환 완료.")
    return data


def _convert_numpy():
    dataset = {}
    dataset["train_img"] = _load_img(data_gz_files["train_img"])
    dataset["train_label"] = _load_label(data_gz_files["train_label"])
    dataset["test_img"] = _load_img(data_gz_files["test_img"])
    dataset["test_label"] = _load_label(data_gz_files["test_label"])
    return dataset


def init_mnist():
    download_mnist()
    dataset = _convert_numpy()
    print("데이터를 pickle 파일로 저장...")
    with open(saved_mnist_file, "wb") as f:
        # -1은 protocol=pickle.HIGHEST_PROTOCOL 의미인 듯...
        pickle.dump(dataset, f, -1)
    print("저장 완료.")


def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1
    return T


def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    """MNIST 데이터셋 읽기
    Parameters
    ----------
    normalize : 이미지 픽셀값 정규화
    one_hot_label : 레이블을 원-핫(one-hot) 배열로
    flatten : 입력 이미지를 1차원 배열로
    Returns
    -------
    (훈련 이미지, 훈련 레이블), (시험 이미지, 시험 레이블)
    """
    if not os.path.exists(saved_mnist_file):
        init_mnist()

    with open(saved_mnist_file, "rb") as f:
        dataset = pickle.load(f)

    if normalize:
        for key in ("train_img", "test_img"):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    if one_hot_label:
        dataset["train_label"] = _change_one_hot_label(dataset["train_label"])
        dataset["test_label"] = _change_one_hot_label(dataset["test_label"])

    if not flatten:
        for key in ("train_img", "test_img"):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return (dataset["train_img"], dataset["train_label"]), (
        dataset["test_img"],
        dataset["test_label"],
    )
