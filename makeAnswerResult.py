import numpy as np
import pandas as pd

def logistic(theta, a, b):
    p = 1.0/(1.0+np.exp(-1.7*a*(theta-b)))
    return p

# main
ITEM = 10
USER = 20

a = np.random.random(ITEM) # 識別力
b = np.random.random(ITEM) # 難易度
theta = np.random.randn(USER) # 特性値

# 確率行列
probability = np.zeros((ITEM, USER))
for i in range(0, ITEM):
    for j in range(0, USER):
        probability[i, j] = logistic(theta[j], a[i], b[i])

probability = np.round(probability, 2)

df = pd.DataFrame(probability)
df.to_csv('probability.csv', index=False, header=None)

# 正誤結果
result = np.zeros((ITEM, USER))
for i in range(0, ITEM):
    for j in range(0, USER):
        if np.random.rand()>probability[i, j]:
            result[i, j] = 1

df = pd.DataFrame(result.astype(np.int))
df.to_csv('result.csv', index=False, header=None)
