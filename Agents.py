import pandas as pd 
import numpy as np 
from Tokenizer import Tokenizer
from Vectorizer import Vectorizer
from Classfier import LGB_Classifer, CountBased_Classifier
from Metrics import Report
import json 
import time 

def train_v2():
    from konlpy.tag import Okt
    from sklearn.feature_extraction.text import TfidfVectorizer

    def tokenizer(raw, pos=["Noun","Alpha","Verb","Number"], stopword=[]):
        return [
            word for word, tag in okt.pos(
                raw, 
                norm=True,   # normalize 그랰ㅋㅋ -> 그래ㅋㅋ
                stem=True    # stemming 바뀌나->바뀌다
                )
                if len(word) > 1 and tag in pos and word not in stopword
        ]


    okt = Okt()
    start_time = time.perf_counter()
    train_df = pd.read_table('./dataset/train.tsv')
    print(f"training samples : {len(train_df)}")
    
    tokenized_time = time.perf_counter()
    
    train_x = train_df['titles']

    labels = train_df['labels']
    vectorizer = TfidfVectorizer(tokenizer=tokenizer)
    vectorizer.fit(train_x)
    vectorized = vectorizer.transform(train_x)

    vectorized_time = time.perf_counter()

    ml_model = LGB_Classifer(model_path="./models")
    ml_model.train(train_x = vectorized,
                        train_y=labels)                        
    ml_model.save_model()


    ml_model.load_model()
    ml_pred=ml_model.predict(vectorized)
    Report(labels, ml_pred, "train_ml_pred_v2")

    
    end_time = time.perf_counter()
    print(f"Training Done at {(end_time - start_time) * 1000} ms seconds")

def train():
    start_time = time.perf_counter()
    train_df = pd.read_table('./dataset/train.tsv')
    print(f"training samples : {len(train_df)}")
    tokenizer = Tokenizer(mode='train', data_path='./dataset/train.tsv')
    train_x = tokenizer.tokenize_dataset(train_df['titles'].values)
    
    tokenized_time = time.perf_counter()
    
    train_df['tokens'] = train_x
    train_df.to_csv("train_test.csv")

    labels = train_df['labels']
    vectorizer = Vectorizer(model_path="./models")
    vectorizer.train(train_x)
    vectorizer.save_model()
    vectorized = vectorizer.transform(train_x)

    vectorized_time = time.perf_counter()

    ml_model = LGB_Classifer(model_path="./models")
    ml_model.train(train_x = vectorized,
                        train_y=labels)                        
    ml_model.save_model()

    stat_model = CountBased_Classifier(model_path="./models")
    stat_model.train(train_x = train_x,
                    train_y = labels)
    stat_model.train_report()
    stat_model.save_table()

    ml_model.load_model()
    ml_pred=ml_model.predict(vectorized)
    Report(labels, ml_pred, "train_ml_pred")

    stat_model.load_table()
    stat_pred = stat_model.predict(train_x)
    Report(labels, stat_pred, "train_stat_pred")
    
    merged_pred = np.add(ml_pred, stat_pred)
    Report(labels, merged_pred, "train_ensamble_pred")

    end_time = time.perf_counter()
    print(f"Training Done at {(end_time - start_time) * 1000} ms seconds")


def test():
    start_time = time.perf_counter()
    test_df = pd.read_table('./dataset/test.tsv')
    tokenizer = Tokenizer(mode='serve', data_path='./models/word_dict.json')
    test_x = tokenizer.tokenize_dataset(test_df['titles'].tolist())
    labels = test_df['labels']
    

    tokenize_time = time.perf_counter()


    vectorizer = Vectorizer(model_path="./models")
    vectorizer.load_model()
    vectorized = vectorizer.transform(test_x)


    vectorize_time = time.perf_counter()


    ml_model = LGB_Classifer(model_path="./models")
    ml_model.load_model()
    ml_pred=ml_model.predict(vectorized)


    ml_model_time = time.perf_counter()
    Report(labels, ml_pred, "test_ml_pred")
    
    stat_model = CountBased_Classifier(model_path="./models")
    stat_model.load_table()
    stat_pred = stat_model.predict(test_x)


    stat_model_time = time.perf_counter()
    Report(labels, stat_pred, "test_stat_pred")
    

    merged_pred = np.add(ml_pred, stat_pred)
    Report(labels, merged_pred, "test_ensamble_pred")

    end_time = time.perf_counter()

    print(f"Testing samples : {len(test_df)}")
    print(f"Testing Done at {(end_time - start_time)*1000} ms seconds")
    print(f"Tokenize Time : {(tokenize_time - start_time)*1000} ms seconds")
    print(f"Vectorized Time : {(vectorize_time- tokenize_time)*1000} ms seconds")
    print(f"ML Inference Time : {(ml_model_time- vectorize_time)*1000} ms seconds")
    print(f"STAT Inference Time : {(stat_model_time- ml_model_time)*1000} ms seconds")        



