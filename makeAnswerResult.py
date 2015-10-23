# IRTのパッケージ(ltm)の動作確認のシミュレーション実験
# IRTにもとづいて人工データを生成する

import numpy as np
import pandas as pd

# ロジスティック関数
def logistic(theta, a, b):
    p = 1.0/(1.0+np.exp(-a*(theta-b)))
    return p

# main
USER = 200
ITEM = 5

# パラメータ設定
#a = np.random.random(ITEM) # 識別力
a = np.ones(ITEM) # 1母数にするために1にする．

#b = np.random.random(ITEM) # 難易度
b = np.array([-6,-1,-1,5,5]) # 手入力して試す．

theta = np.random.randn(USER) * 4 # 特性値. 4倍して範囲を-4から4にする.


# 確率行列
probability = np.zeros((USER, ITEM))
for i in range(0, USER):
    for j in range(0, ITEM):
        probability[i, j] = logistic(theta[i], a[j], b[j])

probability = np.round(probability, 2) # 可読性のため下2桁にする

df = pd.DataFrame(probability)
df.to_csv('probability.csv', index=False, header=None)


# 正誤結果
result = np.zeros((USER, ITEM))
for i in range(0, USER):
    for j in range(0, ITEM):
        if np.random.rand()<probability[i, j]:
            result[i, j] = 1

df = pd.DataFrame(result.astype(np.int))
df.to_csv('result.csv', index=False, header=None)
