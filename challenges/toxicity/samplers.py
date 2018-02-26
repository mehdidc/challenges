import numpy as np

def tfidf():
    rng  = np.random
    params = {
        'max_depth': rng.randint(10, 100),
        'n_estimators': 200,
    }
    code = open('models/tfidf.py').read()
    code = code.format(**params)
    return {
        'codes':{
            'classifier': code,
        },
        'info': params,
    }

def gru():
    rng  = np.random
    params = {
        'nb_units': 100,
        'lr': 1e-3,
        'epochs': 1,
        'batch_size': 64,
    }
    code = open('models/gru.py').read()
    code = code.format(**params)
    return {
        'codes':{
            'classifier': code,
        },
        'info': params,
    }
