# 概要
# 歌詞検索サービス[Uta-Net](https://www.uta-net.com/)から各アーティストの歌詞をスクレイピングするコードです。

# ライブラリのインポート
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

# 保存先のパス
PATH = './data/'

# アーティストの歌詞一覧へのリンク
# - `url_dict`に`{'歌手名': ['URL1', 'URL2', ...]}`の形式でリンクを手動で記述する。
# - 歌詞一覧が2ページ以上にまたがる場合はURLを複数記述することになる。
url_dict = {
    'YOASOBI' : ['https://www.uta-net.com/artist/28370/'],
    'スピッツ' : ['https://www.uta-net.com/artist/1475/0/1/', 'https://www.uta-net.com/artist/1475/0/2/'],
    'Mr.Children': ['https://www.uta-net.com/artist/684/0/1/', 'https://www.uta-net.com/artist/684/0/2/'],
    '米津玄師': ['https://www.uta-net.com/artist/12795/'],
    'SEKAI NO OWARI': ['https://www.uta-net.com/artist/9699/'],
    'あいみょん' : ['https://www.uta-net.com/artist/17598/'],
    'King Gnu': ['https://www.uta-net.com/artist/23343/'],
    'Mrs. GREEN APPLE': ['https://www.uta-net.com/artist/18526/'],
    'Official髭男dism': ['https://www.uta-net.com/artist/18093/'],
    '安室奈美恵': ['https://www.uta-net.com/artist/1822/'],
    'back number': ['https://www.uta-net.com/artist/8613/'],
    'ヨルシカ': ['https://www.uta-net.com/artist/22653/'],
    'BUMP OF CHICKEN': ['https://www.uta-net.com/artist/126/'],
    'ONE OK ROCK': ['https://www.uta-net.com/artist/7063/'],
    'RADWIMPS': ['https://www.uta-net.com/artist/4082/'],
    "B'z": ['https://www.uta-net.com/artist/134/0/1/', 'https://www.uta-net.com/artist/134/0/2/'],
    'ゆず': ['https://www.uta-net.com/artist/1750/0/1/', 'https://www.uta-net.com/artist/1750/0/2/'],
    '嵐': ['https://www.uta-net.com/artist/3891/0/1/', 'https://www.uta-net.com/artist/3891/0/2/'],
    'GreeeeN': ['https://www.uta-net.com/artist/5384/'],
    'サザンオールスターズ': ['https://www.uta-net.com/artist/1395/0/1/', 'https://www.uta-net.com/artist/1395/0/2/'],
    '宇多田ヒカル': ['https://www.uta-net.com/artist/1892/'],
    '星野源': ['https://www.uta-net.com/artist/9867/'],
    'ポルノグラフィティ': ['https://www.uta-net.com/artist/1686/0/1/', 'https://www.uta-net.com/artist/1686/0/2/'],
    'Eve': ['https://www.uta-net.com/artist/20987/'],
    'Ado': ['https://www.uta-net.com/artist/29298/'],
    'BTS': ['https://www.uta-net.com/artist/16377/'],
    '中島みゆき': ['https://www.uta-net.com/artist/3315/0/1/', 'https://www.uta-net.com/artist/3315/0/2/', 'https://www.uta-net.com/artist/3315/0/3/'],
    '緑黄色社会': ['https://www.uta-net.com/artist/22823/'],
    '倖田來未': ['https://www.uta-net.com/artist/2261/0/1/', 'https://www.uta-net.com/artist/2261/0/2/'],
    '優里': ['https://www.uta-net.com/artist/28773/'],
    'いきものがかり': ['https://www.uta-net.com/artist/5580/'],
    'ZARD': ['https://www.uta-net.com/artist/1155/'],
    '椎名林檎': ['https://www.uta-net.com/artist/3361/'],
    '小田和正': ['https://www.uta-net.com/artist/2673/'],
    'TWICE': ['https://www.uta-net.com/artist/21906/'],
    'Perfume': ['https://www.uta-net.com/artist/5555/'],
    'Uru': ['https://www.uta-net.com/artist/20238/'],
    'Superfly': ['https://www.uta-net.com/artist/6895/'],
    'aiko': ['https://www.uta-net.com/artist/39/'],
    'Aimer': ['https://www.uta-net.com/artist/11629/'],
    '三代目 J SOUL BROTHERS': ['https://www.uta-net.com/artist/10539/']
}

# 歌詞を取得
# - 上に記載したリンク先のHTMLを取得し、歌詞を取得する。

#「xxxの歌詞一覧」のHTMLを取得
html_dict = dict()
for artist,urls in tqdm(url_dict.items()):
    html_dict[artist] = []
    for url in urls:
        res = requests.get(url)
        html_dict[artist].append(BeautifulSoup(res.content, 'html.parser'))

# 歌詞を取得
## ＜テキストの処理方法＞
## 全角スペース(\u3000, Webページ上では改行に見える)は「*」に置換
## 上の処理後、「****」は「**」に置換

artist_arr = []
title_arr = []
lyric_arr = []
for artist, htmls in html_dict.items():
    for html in htmls:
        for table in html.find_all('tbody', class_='songlist-table-body'):
            for tr in table.find_all('tr'):
                title = tr.find('span', class_='fw-bold songlist-title pb-1 pb-lg-0').text
                lyric = tr.find('span', class_='d-block pc-utaidashi').text
                lyric = lyric.replace('\u3000','*').replace('****','**')
                artist_arr.append(artist)
                title_arr.append(title)
                lyric_arr.append(lyric)

artist_arr = np.array(artist_arr)
title_arr = np.array(title_arr)
lyric_arr = np.array(lyric_arr)

# CSVを作成し、保存
## lyric_all.csv   :  1行=1曲の歌詞全体
## lyric_block.csv :  1行=1ブロック分の歌詞（**で分割）

df_all = pd.DataFrame({'artist':artist_arr, 'title':title_arr, 'lyric':lyric_arr})
df_all.to_csv(PATH+'lyric_all.csv', index=False)

artist_block_arr = []
title_block_arr = []
block_arr = []
for artist,title,lyric in zip(artist_arr, title_arr, lyric_arr):
    for block in lyric.split('**'):
        artist_block_arr.append(artist)
        title_block_arr.append(title)
        block_arr.append(block)

df_block = pd.DataFrame({'artist':artist_block_arr, 'title':title_block_arr, 'block':block_arr})
df_block.to_csv(PATH+'lyric_block.csv', index=False)

print(df_all.shape, df_block.shape)