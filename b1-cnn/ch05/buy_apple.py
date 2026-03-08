from layer_naive import MulLayer

apple = 100
apple_num = 2
tax = 1.1

# 곱하기 레이어로만 구성된 예제...
# 근데 메서드만 쓸건데 굳이 클래스로 만들어야 하나? - 아...역전파에서 x, y를 곱해줘야 하지...
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# forward - 두 값 넣고 곱한 값 하나 받기
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)

print(price)

# backward - 역전파, 상류 값 하나 넣고, 하류로 갈 값 두 개 받기(곱이라 서로 바뀜)
dprice = 1
# 세금의 미분은 여기 dtax로 끝
dapple_price, dtax = mul_tax_layer.backward(dprice)
# 사과는 다시 가격과 갯수로 구분...
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print(dapple, dapple_num, dtax)
