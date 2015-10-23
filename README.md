# Item Response Theory

## 項目反応理論のRパッケージ(ltm)の動作確認

ltmが正しく動作しているのか検証する必要がある．
そこで，IRTにもとづいて人工データを作成し，そのデータをIRTにかける．事前に決定したパラメータと，プログラムの実行結果が同じになるか調査する．

### 人工データ生成

生成方法を以下に示す．

(1) ユーザ数，問題数を決定する．

(1) a, b, thetaの値を決める．現在のプログラムでは，1母数となっており，aは全て1, bは手動で変化させながら実行確認を行う．thetaは正規乱数 x 4として与えている．

(1) パラメータをロジスティック関数に与えて，確率行列を生成する

(1) 確率行列から正誤データを生成する

人工データはPythonで動作する．2母数ロジスティックモデルまで対応している．

```
$ python makeAnswerResult.py
```

これにより，

```
probability.csv
result.csv
```

が生成される．

### IRTの動作確認

Rのパッケージltmを動かす．

```
> source('irt.r')
```

これにより，グラフが表示される．

人工データを生成するパラメータbである難易度を変更すると，IRTで求めたグラフが変化するので，生成モデルとIRTは関連のある動作をしていると考えられる．

# 課題

IRTの出力値と，人工データのパラメータが一致しない．
IRTの出力結果の解釈ができていない．
ltmパッケージのマニュアルを読む必要がある．
https://cran.r-project.org/web/packages/ltm/ltm.pdf