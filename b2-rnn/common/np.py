from common.config import GPU

if GPU:
    import cupy as np

    # 뭔지 잘 모르겠지만, 메모리 풀 할당자라는 것을 cupy로 변경하는 모양...
    np.cuda.set_allocator(np.cuda.MemoryPool().malloc)
    # 뭔가 예전(8이하)에는 add.at을 호출해 scatter_add를 연결해줘야 했는데, 이게 없어졌다고...
    # 지금은 cupyx에 scatter_add 함수를 사용한다고...
    # np.add.at = np.scatter_add

    # \033[ 터미널 명령 이스케이프, 92m 밝은 녹색, - 60번?, 0m 터미널 속성 초기화...
    print("\033[92m" + "-" * 60 + "\033[0m")
    print(" " * 23 + "\033[92mGPU Mode (cupy)\033[0m")
    print("\033[92m" + "-" * 60 + "\033[0m\n")
else:
    import numpy as np
