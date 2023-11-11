import os
import copy
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
from janome.tokenizer import Tokenizer
from sklearn.metrics import f1_score, accuracy_score

def encode(txt, max_length=200):
    """歌詞の1文字1文字をUnicodeに変換する関数"""
    txt_list = []
    for l in txt:
        txt_line = [ord(x) for x in str(l).strip()]
        txt_line = txt_line[:max_length]
        txt_len = len(txt_line)
        if txt_len < max_length:
            txt_line += ([0] * (max_length - txt_len))
        txt_list.append((txt_line))
    return np.array(txt_list)

class Trainer:
    def __init__(self, n_epochs, batch_size, learning_rate, criterion, opt, gkf, groups, pretrain, device):
        self.batch_size = batch_size
        self.device = device
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.criterion = criterion
        self.opt = opt
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
            optimizer = self.opt(model.parameters(), lr=self.learning_rate)

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
                    loss = self.criterion(outputs, torch.from_numpy(y_va).to(self.device))
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
            filepath = os.path.join(dirname, f'model_fold{fold+1}.pth')
            torch.save(model.state_dict(), filepath)
            print(f'[Saved] score:{self.bst_score[fold]:.4f}  @ {filepath}')

    def save_clf(self, dirname):
        os.makedirs(dirname, exist_ok=True)
        for fold, model in self.bst_model.items():
            filepath = os.path.join(dirname, f'classifier_fold{fold+1}.pth')
            clf_s = OrderedDict(list(model.state_dict().items())[1:])
            torch.save(clf_s, filepath)
            print(f'[Saved] score:{self.bst_score[fold]:.4f}  @ {filepath}')



def predict_one_block(txt, embed, classifier, device):
    """歌詞1ブロックに対する予測を行う関数
    txt[string] : 歌詞1ブロック分

    --> 各アーティストに対する確率値の配列(1次元)
    """
    txt = txt.strip().replace('\n','*').replace('\u3000','*')
    txt_enc = torch.from_numpy(encode([txt]))
    txt_emb = embed(txt_enc.to(device))
    probs = []
    for artist,model_list in classifier.items():
        probs_fold = []
        for model in model_list:
            with torch.no_grad():
                model.eval()
                output = model(txt_emb)
                probs_fold.append(output.softmax(dim=1).cpu().numpy()[0])
        probs.append(np.array(probs_fold).mean(axis=0))
    prob = np.array(probs)[:,1]
    return prob

def predict_some_block(txt_list, embed, classifier, device):
    """歌詞ブロックのリストに対する予測を行う関数
    txt_list[list] : 歌詞1ブロックを1要素とするリスト

    --> 各アーティストに対する確率値の配列。(txt_listの長さ, 全アーティスト数)の2次元配列
    """
    txt_arr = []
    for txt in txt_list:
        txt = txt.replace(' ','')
        txt = txt.strip().replace('\n','*').replace('\u3000','*')
        txt_arr.append(txt)
    txt_enc = torch.from_numpy(encode(txt_arr))
    txt_emb = embed(txt_enc.to(device))
    probs = []
    for artist,model_list in classifier.items():
        probs_fold = []
        for model in model_list:
            with torch.no_grad():
                model.eval()
                output = model(txt_emb)
                probs_fold.append(output.softmax(dim=1).cpu().numpy())
        probs.append(np.array(probs_fold).mean(axis=0))
    probs = np.array(probs)[:,:,1].transpose(1,0)
    return probs

def predict_whole_song(txt_list, embed, classifier, device):
    """1曲のリストに対する予測を行う関数
    txt_list[list] : 1曲の歌詞を1要素とするリスト。ただし1曲の歌詞は'\n\n'によりブロックで分割されていること。

    --> 各アーティストに対する確率値の配列。(txt_listの長さ, 全アーティスト数)の2次元配列
    """
    prob_arr = []
    for txt in txt_list:
        prob_arr.append(predict_some_block(txt.split('\n\n'), embed, classifier, device).mean(axis=0))
    return np.array(prob_arr)

def show_predict_one_block(prob, artists, sort, figsize=(10,3)):
    """予測結果を可視化する関数(歌詞ブロック版)"""
    if sort:
        order = np.argsort(prob)[::-1]
        prob = np.array(prob)[order]
        artists = np.array(artists)[order]

    plt.figure(figsize=figsize)
    plt.bar(artists, prob, color='green', zorder=100)
    for i,p in enumerate(prob):
        plt.text(i, p+0.05, f'{int(p*100)}%', horizontalalignment='center', zorder=100)
    plt.xticks(rotation=90)
    plt.ylim(0,1)
    plt.grid()
    plt.show()

def show_predict_whole_song(prob_arr, artists, sort, raw_txt_arr, figsize=(10,7)):
    """予測結果を可視化する関数(歌詞全体版)"""
    fig,axs = plt.subplots(2,1, figsize=figsize)
    # 予測値の平均
    prob_mean = prob_arr.mean(axis=0)
    if sort:
        order = np.argsort(prob_mean)[::-1]
        prob_mean = np.array(prob_mean)[order]
        artists_sort = np.array(artists)[order]

    axs[0].bar(artists_sort, prob_mean, color='green', zorder=100)
    for i,p in enumerate(prob_mean):
        axs[0].text(i, p+0.05, f'{int(p*100)}%', horizontalalignment='center', zorder=100)
    axs[0].set_xticks(range(len(artists_sort)), labels=artists_sort, rotation=90)
    axs[0].set_ylim(0,1)
    axs[0].grid()
    axs[0].set_title('平均値')

    # 各ブロックの予測値
    prob_norm = prob_arr / np.repeat(prob_arr.sum(axis=1).reshape(-1,1), prob_arr.shape[1], axis=1)
    raw_txt_arr = [raw_txt[:5] for raw_txt in raw_txt_arr]
    df = pd.DataFrame(prob_norm, columns=artists, index=raw_txt_arr)
    df.plot.bar(stacked=True, cmap='tab20c', ax=axs[1], zorder=100)
    axs[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axs[1].set_xticks(range(len(df.index)), labels=df.index, rotation=90)
    axs[1].set_xlabel('')
    axs[1].grid()
    axs[1].legend(ncol=2, bbox_to_anchor=(1, 0.5), loc='center left', fontsize=10)
    axs[1].set_title('各ブロックの予測値')
    plt.tight_layout()
    plt.show()

def wakachi_one_block(txt):
    """日本語テキストを分かち書きする(歌詞ブロック版)"""
    txt = txt.replace('\u3000','')
    tokenizer = Tokenizer()
    tokens = tokenizer.tokenize(txt)
    words = [token.surface for token in tokens]
    txt_split = []
    line = []
    for i,word in enumerate(words):
        if word == '\n':
            txt_split.append(' '.join(line))
            line = []
        else:
            line.append(word)
    txt_split.append(' '.join(line))
    return '\n'.join(txt_split)

def wakachi_some_block(txt):
    """日本語テキストを分かち書きする(歌詞全体版)"""
    wakachi_txt = ''
    for block in txt.split('\n\n'):
        wakachi_txt += wakachi_one_block(block) + '\n\n'
    return wakachi_txt

def highlight(exp, wakachi_txt, artists, sort_by=False):
    """LIMEの結果をハイライト表示"""
    highlighted_text = '<h2><span style="color: black; background-color: rgba(255,128,0,1);">ぽい</span> / <span style="color: black; background-color: rgba(0,128,255,1);">ぽくない</span>判定理由</h2>'

    if sort_by is not False:
        order = np.argsort(sort_by)[::-1]
        artist_labels = np.arange(len(artists))[order]
    else:
        artist_labels = range(len(artists))
    for label in artist_labels:
        words = [word for word, weight in exp.as_list(label)]
        weights = np.array([weight for word, weight in exp.as_list(label)])
        weights_alpha = weights / np.abs(weights).max()
        alpha_dict = {word:alpha for word,alpha in zip(words,weights_alpha)}

        highlighted_text += f'<h3>「{artists[label]}」っぽさ</h3><div style="overflow:auto; max-height:300px; max-width:600px;border: 2px solid black; padding: 20px; box-sizing: border-box"><p style="line-height: 2;">'
        for line in wakachi_txt.split('\n'):
            for word in line.split(' '):
                try:
                    alpha = alpha_dict[word]
                    if alpha > 0:
                        bg_color = (255,128,0,alpha)
                    else:
                        bg_color = (0,128,255,-alpha)
                    highlighted_text += f'<span style="color: black; background-color: rgba{bg_color};">{word}</span> '
                except KeyError:
                    highlighted_text += word
            highlighted_text += '<br>'
        highlighted_text += '</p></div><br>'
    return highlighted_text