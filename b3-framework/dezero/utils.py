import sys

sys.path.append("b3-framework")

from dezero import Variable
import numpy as np
import os
import subprocess
import dezero
from urllib import request
import urllib.request


TMP_DIR = "b3-framework\\tmp"


def _dot_var(v, verbose=False):
    # 변수를 위한 dot 문서 기본 꼴을 문자열로 만들어놓고...
    dot_var = '{} [label="{}", color=orange, style=filled]\n'
    # name 속성의 None 처리...
    name = "" if v.name is None else v.name

    # verbose 선택되어 있다면(거기에 데이터도 있다면) 추가 처리를 하는데...
    if verbose and v.data is not None:
        if v.name is not None:
            name += ": "
        # 이름을 name: shape dtype 형식으로 확장
        name += str(v.shape) + " " + str(v.dtype)
    # id 생성하고 이름과 넣어서 반환
    return dot_var.format(id(v), name)


def _dot_func(f):
    # 역시 함수에 대한 dot 문법 기본 꼴 - 박스로...
    dot_func = '{} [label="{}", color=lightblue, style=filled, shape=box]\n'
    # 함수 이름은 클래스 이름으로...
    txt = dot_func.format(id(f), f.__class__.__name__)

    # 화살표는 함수에서 그리는데...
    dot_edge = "{} -> {}\n"
    # 이 함수로 들어오는 변수들 처리
    for x in f.inputs:
        txt += dot_edge.format(id(x), id(f))
    # 이 함수에서 나가는 변수들 처리
    for y in f.outputs:
        txt += dot_edge.format(id(f), id(y()))
    return txt


def get_dot_graph(output, verbose=True):
    # dot 문법 문자열 만들 것
    txt = ""
    # 함수 목록과 함수 중복 방지를 위한 set
    funcs = []
    seen_set = set()

    # 1. 마지막 변수 dot에 추가
    txt += _dot_var(output, verbose)

    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            seen_set.add(f)

    add_func(output.creator)

    while funcs:
        # 2. 함수 꺼내서 dot에 추가 - 화살표도 여기서 추가
        func = funcs.pop()
        txt += _dot_func(func)
        # 꺼낸 함수에서 앞쪽 입력 변수들 돌면서 dot에 추가
        for x in func.inputs:
            txt += _dot_var(x, verbose)

            # 추가 탐색을 위해 생성자 함수 다시 추가(이전에 등록되지 않았다면)
            if x.creator is not None:
                add_func(x.creator)
    return "digraph g {\n" + txt + "}"


def plot_dot_graph(output, verbose=True, to_file="graph.png"):
    dot_graph = get_dot_graph(output, verbose)

    # tmp_dir = os.path.join(os.path.expanduser("~"), ".dezero")
    # 책과는 다르게 tmp 폴더 만들고 거기로 보내보자...
    if not os.path.exists(TMP_DIR):
        os.mkdir(TMP_DIR)
    graph_path = os.path.join(TMP_DIR, "tmp_graph.dot")

    with open(graph_path, "w") as f:
        f.write(dot_graph)

    extension = os.path.splitext(to_file)[1][1:]
    # 책과는 다르게 그림파일도 tmp 폴더에 보내자...
    to_file = os.path.join(TMP_DIR, to_file)
    cmd = "dot {} -T {} -o {}".format(graph_path, extension, to_file)
    subprocess.run(cmd, shell=True)

    # 주피터 노트북 작업이면 바로 표시 시도...일단 여기선 오류나도 상관없음...
    try:
        from IPython import display

        return display.Image(to_file)
    except:
        pass


# 이건 functions의 Sum을 위한 보조 함수인데...
def reshape_sum_backward(gy, x_shape, axis, keepdims):
    # 입력 x의 원래 shape 차원을 받아두고...(2,3) -> 2차원
    ndim = len(x_shape)
    # axis를 None 또는 튜플 형태로...
    tupled_axis = axis
    if axis is None:
        tupled_axis = None
    elif not isinstance(axis, tuple):
        tupled_axis = (axis,)

    # x가 스칼라가 아니고, axis도 지정되어 있고, 그런데 keepdims는 아닌 경우라..
    if not (ndim == 0 or tupled_axis is None or keepdims):
        # 실제 축인가? axis에서 하나씩 꺼내서 0보다 크면 그대로, 아니면 차원을 더해?
        actual_axis = [a if a >= 0 else a + ndim for a in tupled_axis]
        # 미분 gy의 shape에다 실제 축 원소를 하나씩 더해? 이게 무슨...
        shape = list(gy.shape)
        for a in sorted(actual_axis):
            shape.insert(a, 1)
    else:
        # x가 스칼라거나, axis 없거나, keepdims하라고 하면 그냥 미분 shape 그대로 반환
        shape = gy.shape

    # 정해진 shape 값으로 변환해서 반환...
    gy = gy.reshape(shape)
    return gy


def sum_to(x, shape):
    # shape는 넘파이 브로드캐스트 일어났던, 작은 크기의 배열...
    ndim = len(shape)
    # lead는 이 sum_to 처리를 하고 난 후에 바깥에 몇 차원이 남는지를 계산하는 모양인데...
    # 즉 x의 차원에서 shape의 차원을 빼면 바깥에 남는 차원이 되는 모양인데...
    lead = x.ndim - ndim
    # 그걸 다 벗기기 위해서 0부터 튜플로 만들고 squeeze에서 사용
    lead_axis = tuple(range(lead))

    # 뭔가 원소 수가 1이되는 차원에 대해서만...얼마만에 나오냐에 남는 차원 더하면 합칠 차원이 되는 모양...
    # 이게 절대로 이해가 가질 않는다...
    axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
    y = x.sum(axis=lead_axis + axis, keepdims=True)
    if lead > 0:
        # 그 바깥쪽 남는 차원 벗기기...
        y = y.squeeze(lead_axis)
    return y


def logsumexp(x, axis=1):
    # 이건 오버플로 방지책같고...
    m = x.max(axis=axis, keepdims=True)
    y = x - m
    # y = np.exp(y)를 말하는 것 같고...결과는 exp(x)/exp(m)
    np.exp(y, out=y)
    # 전체 합인데...Sigma(exp(xi))/exp(m) -> 로그 취하면 log(Sigma) - m
    s = y.sum(axis=axis, keepdims=True)
    # 거기에 m 더하면 결국 원래 x의 log(exp sum)이 된다..이런 얘기..
    return m + np.log(s)


def show_progress(block_num, block_size, total_size):
    bar_template = "\r[{}] {:.2f}%"

    downloaded = block_num * block_size
    p = downloaded / total_size * 100
    i = int(downloaded / total_size * 30)
    if p >= 100.0:
        p = 100.0
    if i >= 30:
        i = 30
    bar = "#" * i + "." * (30 - i)
    sys.stdout.write(bar_template.format(bar, p))
    sys.stdout.flush()


def get_file(url, file_name=None):
    if file_name is None:
        file_name = url[url.rfind("/") + 1 :]
    file_path = os.path.join(TMP_DIR, file_name)

    if not os.path.exists(TMP_DIR):
        os.mkdir(TMP_DIR)

    if os.path.exists(file_path):
        return file_path

    print("Downloading:", file_name)
    try:
        urllib.request.urlretrieve(url, file_path, show_progress)
    except (Exception, KeyboardInterrupt) as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise
    print(" Done")

    return file_path


def get_deconv_outsize(size, k, s, p):
    return s * (size - 1) + k - 2 * p


def get_conv_outsize(input_size, kernel_size, stride, pad):
    return (input_size + pad * 2 - kernel_size) // stride + 1


def pair(x):
    # 스칼라 정수면 (x, x)로 아니면 이미 튜플이거나...
    if isinstance(x, int):
        return (x, x)
    elif isinstance(x, tuple):
        assert len(x) == 2
        return x
    else:
        raise ValueError
