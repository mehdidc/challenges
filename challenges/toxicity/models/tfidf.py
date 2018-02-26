from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

class Classifier:

    def __init__(self):
        self.clf = make_pipeline(
            TfidfVectorizer(),
            RandomForestClassifier(max_depth={max_depth}, n_estimators={n_estimators}, n_jobs=-1),
        )
    
    def fit(self, X, y):
        self.clf.fit(X, y)
    
    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

