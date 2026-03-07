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
