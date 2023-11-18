# artist-prediction-xai

## 概要

歌詞を入力すると、どのアーティストの歌詞っぽいかを判定します。

また、モデル説明手法「LIME」を用いて判定根拠を示します。

![image](https://github.com/matsuda-tkm/artist-prediction-xai/assets/101240248/5ad6c0cd-ecb6-4f22-9be9-23fc544d468b)

## 動かし方

### 今すぐ動かす方法
1. `./demo.ipynb` のColabリンクからデモを実行することができます。
    - 各自のGoogle Driveにコピーを作成して実行してください。

### 自分でモデルを学習させて動かす方法

1. `./scrape.py` を実行して歌詞データを収集します。
    - `/data` に収集データが保存されます。
    - `url_dict` を編集/追加することで、収集データをご自分でカスタマイズできます。
1. `./pretrain.py` を実行してモデルの事前学習を行います。
    - `/pretrain` に学習済みモデルが保存されます。
    - `n_epochs`, `batch_size`, `learning_rate` などをカスタマイズできます。
    - `Trainer` クラスは `utils.py` に実装されています。
1. `./train.py`を実行して、ファインチューニングを行います。
    - 各アーティストに対して「歌詞がそのアーティストかどうか」を分類する二値分類器を学習します。
    - `/models` に学習済みモデルが保存されます。
    - `n_epochs`, `batch_size`, `learning_rate` などをカスタマイズできます。
1. `./demo.ipynb` でデモを実行することができます。 
    - `!git clone https://github.com/...`の部分は適宜ご自身のリポジトリに変更してください。

## Requirement
Google Colaboratoryで動かす場合は、以下のライブラリのインストールが必要です。
```
!pip install japanize-matplotlib
!pip install lime
!pip install janome
```