import lightgbm as lgb
import pandas as pd 
import numpy as np 
import joblib
import json
class LGB_Classifer:
    def __init__(self, model_path='/models'):
        self.params = {
            'boosting_type': 'gbdt', # gradient boosted decision tree
            'objective': 'multiclass', # cross entropy loss 
            'num_class': 17,
            'metric': 'multi_logloss',
            'subsample': .9, # will randomly select part of data without resamplin
            'colsample_bytree': .7, #  randomly select part of features on each iteration
            'reg_alpha': .01, # L1 regularization w getting 0
            'reg_lambda': .01, # L2 regularization w getting smaller not 0 
            'min_split_gain': 0.01, # the minimal gain to perform split
            'min_child_weight': 10, # minimal sum hessian in one leaf
            'nthread':-1,
            'verbose':-1,
            'eta':0.05, # learning rate
            'num_iterations': 500, # num of iteration 
            'max_depth' : -1 # branch infinitely
        } 
            
        self.model_path = model_path
        self.model = None

            
    def train(self, train_x, train_y):
        train_data = lgb.Dataset(train_x, label=train_y)
        self.model = lgb.train(self.params,train_data,100)
    
    def predict(self, input):
        prediction = self.model.predict(input)
        return prediction

    def save_model(self):
        joblib.dump(self.model, self.model_path + '/lgb_clf.pkl')
    
    def load_model(self):
        self.model = joblib.load(self.model_path + '/lgb_clf.pkl')

class CountBased_Classifier:
    def __init__(self, top_n=20, model_path='/models'):
        self.n_of_class = 17
        self.corpus_per_class = []
        self.token_per_class = [] # tf
        self.total_token_freq = {} # df
        self.class_top_token = {}    
        self.top_n = top_n
        self.model_path = model_path
    def train(self, train_x, train_y):
        class_token = {}

        for x, y in zip(train_x, train_y):
            for token in str(x).split(' '):
                try:
                    class_token[y][token] += 1
                except:
                    if y not in class_token.keys():
                        class_token[y] = dict()
                    class_token[y][token] = 1

                try:
                    self.total_token_freq[token][y] += 1
                except:
                    self.total_token_freq[token] = [0] * self.n_of_class
                    self.total_token_freq[token][y] = 1
        
        for c in range(self.n_of_class):
            top_temp = sorted(class_token[c].items(), key=(lambda x:x[1]), reverse=True)
            self.class_top_token[c] = {token:freq for token, freq in top_temp[0:self.top_n]}

        for k, v in self.total_token_freq.items():
            total = sum(v)
            self.total_token_freq[k] = np.true_divide(v, sum(v)).tolist()


    def train_report(self):
        for c in range(self.n_of_class):
            print("***** CountBased_Classifier Training Report *****")
            print(f"{c} class's token info *****")
            for t in self.class_top_token[c]:
                print(f"Token - {t},\t {self.total_token_freq[t]}")
            print("*************************************************")

    def predict(self, train_x):
        predicts = []
        
        for tokens in train_x:
            predict = []

            for token in str(tokens).split(' '):

                for c in range(self.n_of_class):
                    #print(self.class_top_token[str(c)], token)
                    if(token in self.class_top_token[str(c)]):
                        #print(token)
                        predict.append(self.total_token_freq[token])

            #print(tokens)
            if(len(predict) == 0):
                predict = np.array([0] * self.n_of_class)
            else:
                predict = np.array(predict)
                #print(predict.shape, predict)            
                predict = np.average(predict, axis=0)
            
            #print(predict.shape, predict)

            predicts.append(predict)
            #print(predicts.shape, predicts)
            
        #print(tokens, np.sum(predicts, axis=1))
        return np.array(predicts)


    def save_table(self):
        with open(self.model_path + '/class_top_token.json', 'w') as file:
            json.dump(self.class_top_token, file)
        with open(self.model_path + '/total_token_freq.json', 'w') as file:
            json.dump(self.total_token_freq, file)    
        
    def load_table(self):
        with open(self.model_path + '/class_top_token.json', 'r') as file:
            self.class_top_token = json.load(file)
        with open(self.model_path + '/total_token_freq.json', 'r') as file:
            self.total_token_freq = json.load(file)