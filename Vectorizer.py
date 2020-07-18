import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

class Vectorizer:
    def __init__(self, model_path='/models'):
        self.vectorizer = TfidfVectorizer()
        self.model_path = model_path
    def train(self, train_x):
        self.vectorizer.fit(train_x)
    def transform(self, x):
        return self.vectorizer.transform(x).toarray()
    def get_feature_info(self):
        return self.vectorizer.get_feature_names()
    def save_model(self):
        joblib.dump(self.vectorizer, self.model_path + '/tfidf.pkl')        
    def load_model(self):
        self.vectorizer = joblib.load(self.model_path + '/tfidf.pkl')