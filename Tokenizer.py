import pandas as pd 
from konlpy.tag import Okt
from soynlp.word import WordExtractor
from soynlp.tokenizer import RegexTokenizer, LTokenizer, MaxScoreTokenizer
import spacy # spacy 가 nltk와 비교해서 사용성이 더 좋다고 한다
import mecab 
import math
import re 
import numpy as np
import json 


class Tokenizer(object):
    def __init__(self, mode = 'serve', data_path='./dataset/train.tsv'):
        self.mode = mode
        self.data_path = data_path
        self.train_corpus = None
        if(self.mode == 'train'):
            dataset = pd.read_table(self.data_path)
            dataset = dataset[['titles', 'labels']]
            self.train_corpus = dataset['titles'].tolist()

        # python -m spacy download en
        # python -m spacy download en_core_web_sm
        # python -m spacy link en_core_web_sm en

        self.tokenizer = self.soynlp_tokenizer()
        self.nlp = spacy.load('en')         
        self.mecab = mecab.MeCab()
        self.okt = Okt()

        #train_dataset['tokenized_title'] = train_dataset['titles'].map(self.tokenize)
        #train_dataset['merged_title'] = train_dataset['tokenized_title'].map(self.read_arr_to_str)
        #train_dataset.to_csv('train_test.csv', index=False)

    def tokenize_dataset(self, inputs):
        #print("tokenize dataset")
        res = []
        for i in inputs:
            res.append(' '.join(self.tokenize(i)))        
        return np.array(res)
        

    def soynlp_tokenizer(self):
        def word_score(score): return (score.cohesion_forward * math.exp(score.right_branching_entropy))

        if self.mode == 'serve':
            with open(self.data_path, 'r') as file:
                word_score_dict = json.load(file)
        elif self.mode == 'train':
            word_extractor = WordExtractor()
            word_extractor.train(self.train_corpus)
            words = word_extractor.extract()
            word_score_dict = { word:word_score(score) for word, score, in words.items()}

            with open('./models/word_dict.json', 'w') as file:
                json.dump(word_score_dict, file)
        else:
            pass
        
        tokenizer = MaxScoreTokenizer(scores=word_score_dict)
        return tokenizer

    def split_kor_eng(self, input):
        eng = ''
        kor = ''

        input = re.sub(f"r'^_[-+]?([1-9]\d*|0)$'", " ", input)

        for c in input:
            # ord 는 char의 ascii 값을 돌려주는 함수이다.
            if( ord('가') <= ord(c) <= ord('힣')):
                kor += c
            elif( ord('a') <= ord(c.lower()) <= ord('z')):
                eng += c
            else:
                kor += ' '
                eng += ' '
        return eng, kor

    def tokenize(self, input):
        eng, kor = self.split_kor_eng(input)
        
        kor_token = self.mecab.morphs(kor)
        kor_token += self.tokenizer.tokenize(input)
        kor_token += self.okt.morphs(kor)
        
        eng_token = [token.text for token in self.nlp(eng)]

        kor_token = [x.strip() for x in kor_token]
        kor_token = [x for x in kor_token if len(x) > 1]
        eng_token = [x.strip() for x in eng_token]
        eng_token = [x for x in eng_token if len(x) > 1]
        return list(set(kor_token + eng_token))

def test():
    train_df = pd.read_table('./dataset/train.tsv')
    tokenizer = Tokenizer(mode='train', data_path='./dataset/train.tsv')
    res = tokenizer.tokenize_dataset(train_df['titles'])
    print(res)


    test_df = pd.read_table('./dataset/test.tsv')
    res = tokenizer.tokenize_dataset(test_df['titles'])
    print(res)

    test_df = pd.read_table('./dataset/test.tsv')
    new_tokenizer = Tokenizer(mode='serve', data_path='./models/word_dict.json')
    res = new_tokenizer.tokenize_dataset(test_df['titles'])
    print(res)

