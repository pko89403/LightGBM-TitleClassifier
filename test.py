import pandas as pd
pd.options.mode.chained_assignment = None

import numpy as np
np.random.seed(0)
import time 
from konlpy.tag import Okt
okt = Okt()

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

# tokenizer : 문장에서 색인어 추출을 위해 명사,동사,알파벳,숫자 정도의 단어만 뽑아서 normalization, stemming 처리하도록 함
def tokenizer(raw, pos=["Noun","Alpha","Verb","Number"], stopword=[]):
    return [
        word for word, tag in okt.pos(
            raw, 
            norm=True,   # normalize 그랰ㅋㅋ -> 그래ㅋㅋ
            stem=True    # stemming 바뀌나->바뀌다
            )
            if len(word) > 1 and tag in pos and word not in stopword
        ]

# 테스트 문장
train_df = pd.read_table('./dataset/train.tsv')['titles']
test_df = pd.read_table('./dataset/test.tsv')['titles']
print(f"training samples : {len(train_df)}")

vectorize = CountVectorizer(
    tokenizer=tokenizer,
    min_df=2
)

X = vectorize.fit(train_df)

