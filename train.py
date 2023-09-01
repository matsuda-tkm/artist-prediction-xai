# 概要
# 歌詞からアーティスト名を予測する分類モデルを学習させるコードです。
# テキストを文字単位で分割して入力する「CharacterCNN」を採用しています。
# 論文→https://proceedings.neurips.cc/paper_files/paper/2015/file/250cf8b51c773f3f8dc8b4be867a9a02-Paper.pdf

# ライブラリのインポート
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from tqdm import tqdm
import copy
import os
import json
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split, GroupKFold

# 保存先のパス
PATH = './'

# 歌詞CSVの読み込み
df = pd.read_csv(PATH+'data/lyric_block.csv')
print(df.artist.unique())

# アーティストを選択
# - `artists`に分類先のアーティスト名を格納する。

# 分類対象のアーティストを選択
artists = ['あいみょん', 'スピッツ', '星野源', 'YOASOBI']
df_sub = df[df['artist'].isin(artists)]

# アーティスト名を整数化(idとは別。ラベル用)
artist_to_label = {artist:i for i,artist in enumerate(artists)}

# 各アーティストのデータ数を棒グラフで可視化
plt.figure(figsize=(10,3))
df_sub['artist'].value_counts().plot(kind='bar')
plt.show()


# 前処理
# - 各文字をUnicodeに変換
# - 特徴量をXへ、ラベルをyへ格納

def encode(txt, max_length=200):
    """
    歌詞の1文字1文字をUnicodeに変換する関数

    Parameters:
        txt (iterable of str): Text or iterable of texts to be encoded.
        max_length (int, optional): Maximum length of encoded sequence. Default is 200.

    Returns:
        numpy.ndarray: Encoded sequence(s) as a NumPy array.

    Notes:
        - Each character in the input text(s) is converted to its corresponding Unicode code point.
        - The resulting encoded sequence(s) are padded or truncated to match the specified maximum length.
        - If the input text(s) are shorter than the maximum length, the remaining elements are filled with zeros.

    Example:
        txt = ["Hello, world!"]
        encoded_txt = encode(txt, max_length=10)
        print(encoded_txt)
        # Output: [[ 72 101 108 108 111  44  32 119 111 114]]
    """
    txt_list = []
    for l in txt:
        txt_line = [ord(x) for x in str(l).strip()]
        txt_line = txt_line[:max_length]
        txt_len = len(txt_line)
        if txt_len < max_length:
            txt_line += ([0] * (max_length - txt_len))
        txt_list.append((txt_line))
    return np.array(txt_list)


# 説明変数X
X = encode(df_sub['block'])
# 目的変数y
y = df_sub['artist'].map(artist_to_label).values
print(X.shape, y.shape)

# ネットワークの定義
class CharacterCNN(nn.Module):
    def __init__(self, num_classes ,embed_size=128, max_length=200, filter_sizes=(2, 3, 4, 5), filter_num=64):
        super().__init__()
        self.params = {'num_classes': num_classes ,'embed_size':embed_size, 'max_length':max_length, 'filter_sizes':filter_sizes, 'filter_num':filter_num}
        self.embed_size = embed_size
        self.max_length = max_length
        self.filter_sizes = filter_sizes
        self.filter_num = filter_num

        self.embedding = nn.Embedding(0xffff, embed_size)
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(embed_size, filter_num, filter_size) for filter_size in filter_sizes
        ])
        self.fc1 = nn.Linear(filter_num * len(filter_sizes), 64)
        self.batch_norm = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        embedded = self.embedding(x).transpose(1,2)

        conv_outputs = []
        for conv_layer in self.conv_layers:
            conv_output = F.relu(conv_layer(embedded))
            pooled = F.max_pool1d(conv_output, conv_output.size(2)).squeeze(2)
            conv_outputs.append(pooled)

        convs_merged = torch.cat(conv_outputs, dim=1)
        fc1_output = F.relu(self.fc1(convs_merged))
        bn_output = self.batch_norm(fc1_output)
        do_output = self.dropout(bn_output)
        fc2_output = self.fc2(do_output)
        return fc2_output

# 学習
# - `n_epochs`, `batch_size`, `learning_rate`, `optimizer`などを設定する。
# - 同じ歌の歌詞が、学習用データと評価用データへ混在しないようにGroupKFold。（1曲の中で同じ歌詞が繰り返し登場するため）
# - 各foldのBestモデルを保存する。

n_epochs = 10          # エポック数
batch_size = 256       # バッチサイズ
learning_rate = 0.001  # 学習率
n_splits = 5           # GroupKFoldの分割数
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

criterion = nn.CrossEntropyLoss()
gkf = GroupKFold(n_splits)  # GroupKfold

now = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups=df_sub['title'])):
    # 学習データと評価用データに分割
    print(f'-----Fold{fold+1}/{n_splits}-----')
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    n_iter = len(y_tr)//batch_size

    bst_model = None
    bst_score = np.inf
    model = CharacterCNN(num_classes=len(artists), embed_size=128, filter_sizes=(2,3,4,5), filter_num=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 学習
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        random_idx = np.random.permutation(len(y_tr))
        for i in range(n_iter):
            X_batch = torch.from_numpy(X_tr[random_idx[batch_size*i:batch_size*(i+1)]]).to(device)
            y_batch = torch.from_numpy(y_tr[random_idx[batch_size*i:batch_size*(i+1)]]).to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(dim=1)
            total_correct += (predicted == y_batch).sum().item()

        train_loss = total_loss / n_iter
        train_acc = total_correct / (batch_size*n_iter)

        # 評価
        model.eval()
        with torch.no_grad():
            outputs = model(torch.from_numpy(X_va).to(device))
            loss = criterion(outputs, torch.from_numpy(y_va).to(device))
            valid_loss = loss.item()
            _, predicted = outputs.max(dim=1)
            valid_acc = (predicted == torch.from_numpy(y_va).to(device)).sum().item() / len(y_va)
        print(f'Epoch[{epoch+1}/{n_epochs}], TrainLoss: {train_loss:.4f}, ValidLoss: {valid_loss:.4f}, TrainAcc: {train_acc*100:.4f}%, ValidAcc: {valid_acc*100:.4f}%')

        # best更新処理
        if valid_loss < bst_score:
            bst_model = copy.deepcopy(model)
            bst_score = valid_loss
            bst_epoch = epoch + 1

    # ベストモデルを保存
    os.makedirs(PATH+f'models/model_{now}', exist_ok=True)
    torch.save(bst_model.cpu(), PATH+f'models/model_{now}/model_fold{fold+1}.pth')

    # モデルの情報をjsonに記録
    if fold == 0:
        data = dict()
        data['ClassNames'] = artists

    bst_info = dict()
    bst_info['TrainSize'] = len(y_tr)
    bst_info['ValidSize'] = len(y_va)
    bst_info['Params'] = model.params
    bst_info['BatchSize'] = batch_size
    bst_info['LearningRate'] = learning_rate
    bst_info['Optimizer'] = str(type(optimizer))
    bst_info['Epoch'] = bst_epoch
    bst_info['TrainLoss'] = train_loss
    bst_info['ValidLoss'] = valid_loss
    bst_info['TrainAcc'] = train_acc * 100
    bst_info['ValidAcc'] = valid_acc * 100

    data[f'Fold{fold+1}'] = bst_info

with open(PATH + f'models/model_{now}/info.json', 'w') as f:
    json.dump(data, f)