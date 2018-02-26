from sklearn.metrics import roc_auc_score

import pandas as pd
import numpy as np

problem = {
    'name': 'toxicity',
    'workflow': 'miniramp.workflows.classification',
    'workflow_options':{
        'final_model_strategy': 'bagging'
    },
    'data': 'challenges.toxicity.data',
    'validation' : {
        'name': 'miniramp.validation.kfold',
        'params':{
            'n_splits': 5,
            'shuffle': True,
            'random_state': 42,
        }
    },
    'scores' : [
        'challenges.toxicity.avg_auc',
    ],
}

def data():
    return _load_train('data/train.csv')

def avg_auc(clf, X, y):
    aucs = []
    y_pred = clf.predict_proba(X)
    for i in range(y.shape[1]):
        aucs.append(roc_auc_score(y[:, i], y_pred[i][:, 1]))
    return np.mean(aucs)

def _load_train(filename):
    df = pd.read_csv(filename)
    X = df['comment_text']
    y = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]
    X = X.values
    y = y.values
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    nb_train = int(len(X) * 0.5)
    X_train, y_train = X[0:nb_train], y[0:nb_train]
    X_test, y_test = X[nb_train:], y[nb_train:]
    return (X_train, y_train), (X_test, y_test)
