# artist-prediction-xai

## 概要

歌詞を入力すると、どのアーティストの歌詞っぽいかを判定します。

また、モデル説明手法「LIME」を用いて判定根拠を示します。

## 動かし方

1. `./scrape.py`または`./notebooks/scrape.ipynb`を使って歌詞データを収集します。
2. `./train.py`または`./notebooks/train.ipynb`を使ってモデルの学習を行います。
3. `./notebooks/demo.ipynb`でデモを実行することができます。

## Requirement
Google Colaboratoryで動かす場合は、以下のライブラリのインストールが必要です。
```
!pip install japanize-matplotlib
!pip install lime
!pip install janome
```