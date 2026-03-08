import numpy as np

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    # 편향이 문턱값이고 단순히 이걸 낮추면 OR
    b = -0.2
    # 벡터 곱, 그리고 합산
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
    
if __name__ == "__main__":
    for xs in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        result = OR(xs[0], xs[1])
        print(str(xs) + " -> " + str(result))