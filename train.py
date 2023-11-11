import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold

from network import CharacterCNN
from utils import encode, Trainer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# 保存先のパス
PATH = '.'

# 歌詞CSVの読み込み
df = pd.read_csv(os.path.join(PATH,'data/lyric_block.csv'))

# 事前学習重みを読み込む
weight_path = os.path.join(PATH, 'pretrain/model_fold1.pth')
model = CharacterCNN(num_classes=len(artists), embed_size=256, filter_sizes=(2,3,4,5), filter_num=64)
model.load_state_dict(torch.load(weight_path))

# Embeddingのみweightを固定
for param in model.parameters():
    param.requires_grad = False
    break

# 出力層を変更
model.fc2 = nn.Linear(64, 2, bias=True)

n_epochs = 20          # エポック数
batch_size = 128       # バッチサイズ
learning_rate = 0.001  # 学習率
n_splits = 5           # GroupKFoldの分割数

criterion = nn.CrossEntropyLoss()
gkf = GroupKFold(n_splits)  # GroupKfold

for artist in artists:
    df_artist = df[df['artist'] == artist]
    df_other = df[df['artist'] != artist].sample(len(df_artist))
    df_sub = pd.concat([df_artist, df_other])

    X = encode(df_sub['block'])
    y = df_sub['artist'].map({artist:1}).fillna(0).values.astype('int')

    train = Trainer(n_epochs, batch_size, learning_rate, criterion, gkf, df_sub['title'], False, device)
    train.set_model(model, model.state_dict())
    train.train(X, y, 10)
    train.save_clf(os.path.join(PATH, f'models/{artist}'))