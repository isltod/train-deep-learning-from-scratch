import numpy as np

# 이런 표본이 이런 확률로 나올 때
x = np.array([1, 2, 3])
pi = np.array([0.1, 0.1, 0.8])

# 진짜 기댓값은 이런데...
e = np.sum(x * pi)
print(e)

# 그걸 시행 100번의 몬테카를로법으로 구해보면 이렇고...
n = 100
samples = []
for _ in range(n):
    s = np.random.choice(x, p=pi)
    samples.append(s)
mean = np.mean(samples)
var = np.var(samples)
print("몬테카를로법: {:.2f} (분산: {:.2f})".format(np.mean(samples), np.var(samples)))

# importance sampling으로 구하면...전혀 다른 확률 분포가 있고...
# b = np.array([1 / 3, 1 / 3, 1 / 3])
b = np.array([0.2, 0.2, 0.6])
samples = []
# ρ 계산할 때 b 인덱스가 필요하다고 굳이 b 인덱스로 돌려야 하나? 어차피 x가 나와야 말이되는데...
idx = np.arange(len(b))
for _ in range(n):
    i = np.random.choice(idx, p=b)
    s = x[i]
    # 182쪽 식 5.7 계산
    rho = pi[i] / b[i]
    samples.append(rho * s)
mean = np.mean(samples)
var = np.var(samples)
# 이러면 평균도 더 틀리고 분산이 아주 커진다...는 문제가...
print("중요도 샘플링: {:.2f} (분산: {:.2f})".format(np.mean(samples), np.var(samples)))
