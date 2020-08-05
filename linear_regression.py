import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Read_value = pd.read_csv('./lib/train.csv') # 불러올 변량 파일이 들은 위치로 설정.
x_regressor = Read_value['x']
y_response = Read_value['y']

# x, y값의 평균
x_avg = np.mean(x_regressor)
y_avg = np.mean(y_response)

# x,y의 편차의 제곱의 합
Sxx = np.sum((x_regressor-x_avg)**2)
Syy = np.sum((y_response-y_avg)**2)
print(Sxx)
print(Syy)

#  x,y 편차 곱의 합
Sxy = np.sum((x_regressor-x_avg) * (y_response-y_avg))
print(Sxy)

m = Sxy/Sxx
b = y_avg - x_avg * m

print("기울기 :",m)
print("y절편 :",b)

# matplotlib 그래프
x = np.arange(0, 75, 1)
y = [(m*num + b) for num in x]

plt.plot(x, y, c = "b")
plt.scatter(x_regressor, y_response, c = "r")
plt.xlim([-100, 200])
plt.ylim([-100, 300])
plt.draw()
plt.show()
plt.clf()
