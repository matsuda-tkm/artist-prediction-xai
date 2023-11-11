import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold

from network import CharacterCNN
from utils import encode, Trainer

# 保存先のパス
PATH = '.'

# 歌詞CSVの読み込み
df = pd.read_csv(os.path.join(PATH,'data/lyric_block.csv'))
print(df.artist.unique())

# 分類先アーティスト
artists = list(df.artist.unique())
artist_to_label = {artist:i for i,artist in enumerate(artists)}

# 説明変数X
X = encode(df['block'])
# 目的変数y
y = df['artist'].map(artist_to_label).values

# エポック数
n_epochs = 10
# バッチサイズ
batch_size = 512
# 学習率
learning_rate = 0.001
# GroupKFoldの分割数
n_splits = 5
# 損失関数
criterion = nn.CrossEntropyLoss()

gkf = GroupKFold(n_splits)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(0)
model = CharacterCNN(num_classes=len(artists), embed_size=256, filter_sizes=(2,3,4,5), filter_num=64)

pretrain = Trainer(n_epochs, batch_size, learning_rate, criterion, torch.optim.Adam, gkf, df['title'], True, device)
pretrain.set_model(model, model.state_dict())
pretrain.train(X, y)
pretrain.save_all(os.path.join(PATH, 'pretrain'))