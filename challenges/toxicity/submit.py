import pandas as pd
import cloudpickle
from clize import run

def submit(filename, *, out='out.csv'):
    with open(filename, 'rb') as fd:
        models = cloudpickle.load(fd)
    clf = models['final']
    df = pd.read_csv('data/test.csv')
    
    X = df['comment_text'].values
    ids = df['id'].values
    col = {}
    col['id'] = ids
    ypred = clf.predict_proba(X)
    col['toxic'] = ypred[0][:, 1]
    col['severe_toxic'] = ypred[1][:, 1]
    col['obscene'] = ypred[2][:, 1]
    col['threat'] = ypred[3][:, 1]
    col['insult'] = ypred[4][:, 1]
    col['identity_hate'] = ypred[5][:, 1]
    df = pd.DataFrame(col)
    df.to_csv(out, index=False)

if __name__ == '__main__':
    run(submit)
