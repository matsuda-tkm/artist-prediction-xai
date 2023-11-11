import pandas as pd
import numpy as np
import copy
import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, accuracy_score

# 保存先のパス
PATH = '.'

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

class CharacterCNN(nn.Module):
    def __init__(self, num_classes, embed_size=256, filter_sizes=(2, 3, 4, 5), filter_num=64):
        super().__init__()
        self.embed_size = embed_size
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

class Trainer:
    def __init__(self, n_epochs, batch_size, learning_rate, criterion, gkf, groups, pretrain, device):
        self.batch_size = batch_size
        self.device = device
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.criterion = criterion
        self.gkf = gkf
        self.groups = groups
        self.pretrain = pretrain
        self.n_splits = gkf.get_n_splits()
        self.bst_model, self.bst_score = dict(), dict()

    def set_model(self, network, state_dict):
        self.network = network
        self.state_dict = copy.deepcopy(state_dict)

    def reset_model(self):
        model = self.network
        model.load_state_dict(self.state_dict)
        return model

    def train(self, X, y, verbose=1):
        for fold, (tr_idx, va_idx) in enumerate(self.gkf.split(X, y, groups=self.groups)):
            # 学習データと評価用データに分割
            print(f'-----Fold{fold+1}/{self.n_splits}-----')
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]
            n_iter = len(y_tr) // self.batch_size

            model = self.reset_model().to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

            self.bst_model[fold] = None
            self.bst_score[fold] = -np.inf

            # 学習
            for epoch in range(self.n_epochs):
                model.train()
                total_loss = 0.0
                total_correct = 0
                random_idx = np.random.permutation(len(y_tr))
                for i in range(n_iter):
                    X_batch = torch.from_numpy(X_tr[random_idx[self.batch_size*i:self.batch_size*(i+1)]]).to(self.device)
                    y_batch = torch.from_numpy(y_tr[random_idx[self.batch_size*i:self.batch_size*(i+1)]]).to(self.device)

                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    loss = self.criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    _, predicted = outputs.max(dim=1)
                    total_correct += (predicted == y_batch).sum().item()

                train_loss = total_loss / n_iter
                train_acc = total_correct / (self.batch_size*n_iter)

                # 評価
                model.eval()
                with torch.no_grad():
                    outputs = model(torch.from_numpy(X_va).to(self.device))
                    loss = criterion(outputs, torch.from_numpy(y_va).to(self.device))
                    valid_loss = loss.item()
                    _, predicted = outputs.max(dim=1)
                    valid_acc = accuracy_score(y_va, predicted.cpu().numpy())
                    valid_f1 = f1_score(y_va, predicted.cpu().numpy(), average='macro')
                if (epoch+1) % verbose == 0:
                    print(f'Epoch[{epoch+1}/{self.n_epochs}], TrainLoss: {train_loss:.4f}, ValidLoss: {valid_loss:.4f}, TrainAcc: {train_acc*100:.4f}%, ValidAcc: {valid_acc*100:.4f}%, ValidF1: {valid_f1:.4f}')

                # best更新処理
                if valid_f1 > self.bst_score[fold]:
                    self.bst_model[fold] = copy.deepcopy(model)
                    self.bst_score[fold] = valid_f1

            if self.pretrain:
                break

    def save_all(self, dirname):
        os.makedirs(dirname, exist_ok=True)
        for fold, model in self.bst_model.items():
            filepath = dirname + '/' + f'model_fold{fold+1}.pth'
            torch.save(model.state_dict(), filepath)
            print(f'[Saved] score:{self.bst_score[fold]:.4f}  @ {filepath}')

    def save_clf(self, dirname):
        os.makedirs(dirname, exist_ok=True)
        for fold, model in self.bst_model.items():
            filepath = dirname + '/' + f'classifier_fold{fold+1}.pth'
            clf_s = OrderedDict(list(model.state_dict().items())[1:])
            torch.save(clf_s, filepath)
            print(f'[Saved] score:{self.bst_score[fold]:.4f}  @ {filepath}')

# 歌詞CSVの読み込み
df = pd.read_csv(PATH + '/data/lyric_block.csv')
print(df.artist.unique())

# 分類先アーティスト
artists = list(df.artist.unique())
artist_to_label = {artist:i for i,artist in enumerate(artists)}

# 説明変数X
X = encode(df['block'])
# 目的変数y
y = df['artist'].map(artist_to_label).values


n_epochs = 20          # エポック数
batch_size = 512       # バッチサイズ
learning_rate = 0.001  # 学習率
n_splits = 5           # GroupKFoldの分割数
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

criterion = nn.CrossEntropyLoss()
gkf = GroupKFold(n_splits,)  # GroupKfold

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

pretrain = Trainer(n_epochs, batch_size, learning_rate, criterion, gkf, df['title'], True, device)
pretrain.set_model(model, model.state_dict())
pretrain.train(X, y)

pretrain.save_all(PATH + '/pretrain')


# 事前学習重みを読み込む
weight_path = PATH + '/pretrain/model_fold1.pth'
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
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

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
    train.save_clf(PATH + f'/models/{artist}')