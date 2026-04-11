V = {"L1": 0, "L2": 0}

cnt = 0
while True:
    # 중간 더하기는 a에 대한 시그마
    t = 0.5 * (-1 + 0.9 * V["L1"]) + 0.5 * (1 + 0.9 * V["L2"])
    # 이전 VL1과 요번 VL1 차를 먼저 구하고 이번 값으로 업데이트
    delta = abs(t - V["L1"])
    V["L1"] = t

    t = 0.5 * (0 + 0.9 * V["L1"]) + 0.5 * (-1 + 0.9 * V["L2"])
    # 갱신된 값의 최대값...
    delta = max(delta, abs(t - V["L2"]))
    V["L2"] = t

    cnt += 1
    if delta < 0.0001:
        print(V)
        print("갱신 횟수: ", cnt)
        break
