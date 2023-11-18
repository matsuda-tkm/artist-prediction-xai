import os
import streamlit as st
import torch
import numpy as np
import pandas as pd
from collections import OrderedDict
from network import CharacterCNN, CharacterCNNClassifier, CharacterCNNEmbedding
from utils import predict_some_block
parent = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

state_dict_emb = torch.load(os.path.join(parent, 'pretrain', 'model_fold1.pth'), map_location=device)
state_dict_emb = OrderedDict(list(state_dict_emb.items())[0:1])
embed = CharacterCNNEmbedding().to(device)
embed.load_state_dict(state_dict_emb)

classifier = dict()
for artist in os.listdir(os.path.join(parent, 'models')):
    model_list = []
    for file in os.listdir(os.path.join(parent, 'models', artist)):
        state_dict = torch.load(os.path.join(parent, 'models', artist, file), map_location=device)
        clf = CharacterCNNClassifier(2).to(device)
        clf.load_state_dict(state_dict)
        model_list.append(clf)
    classifier[artist] = model_list

artists = list(classifier.keys())

st.title('その歌詞はどのアーティストっぽい？')

# TEXT BOX
text = st.text_area('歌詞を入力してください。歌詞全体を入力するときは、ブロックごとに1行空けるようにしてください。', height=200)
text_list = text.split('\n\n')

# PREDICT
if text == '':
    prob = np.zeros(len(artists))
    st.error('歌詞を入力してください！')
else:
    with st.spinner('Inference...'):
        prob = predict_some_block(text_list, embed, classifier, device)
        prob = prob.mean(axis=0)
    st.success(f'その歌詞は「{artists[np.argmax(prob)]}」っぽい！')

# RESULT
df = pd.DataFrame({'artist':artists,'probability':prob*100})
df.sort_values('probability', ascending=False, inplace=True)
df.reset_index(drop=True, inplace=True)
df.set_index('artist', inplace=True)

if text != '':
    # TOP3
    cols = st.columns(3)
    for i, col in enumerate(cols):
        col.metric(f'{i+1}位', df.index[i], f'{df.probability[i]:.1f}%')

    # BAR CHART
    st.bar_chart(df)
    st.write(df)