# 概要

NF 展示用に作成した NMIST の GUI アプリケーションです。

マウスで書いた数字を認識します。

モデルの作成と学習もできます。

# 使い方

python3 で次のコマンドを実行します。

```
python mnist_gui.py
```

次のパッケージが必要です。pip のパッケージ名で表記

- numpy (行列計算に利用)
- matplotlib (画像の表示に利用)
- tensorflow
  (ニューラルネットワークのライブラリとして利用。keras で利用できるものであれば他の物でもよい。)
- keras (tensorflow のラッパーとして利用。
  "image_data_format": "channels_last" のみに対応。)
- scikit-learn (学習データの作成。モデルの評価に利用。)
- h5py (モデルの保存、読み込みに利用)
- PyQt5 (GUI のライブラリ)

# モデルの作成のやり方

1. Model Editor タブの中の、「追加」ボタンを押すことで層が追加されます。

2. 層の設定が終わったら「モデルをコンパイル」ボタンを押します。

3. 「エディターからモデルをロード」ボタンで、モデルがセットされます。

4. セットしたモデルで、手書き文字認識や、「学習開始」が行えます。

# 参考にしたサイト

PyQt5とpython3によるGUIプログラミング
https://qiita.com/kenasman/items/471b9930c0345562cbbf

Qt Documentation
http://doc.qt.io/
