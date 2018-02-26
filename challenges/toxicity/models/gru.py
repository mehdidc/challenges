import numpy as np
import re
import uuid
from sklearn.metrics import roc_auc_score
from gensim.models.wrappers import FastText
from keras.layers import *
from keras.optimizers import Adam
from keras.models import Model

ft = FastText.load_fasttext_format('wiki.en.bin')
#ft = None
max_length = 100
nb_outputs = 6
vec_size = 300
batch_size = {batch_size}


def normalize(s):
    """
    Given a text, cleans and normalizes it. Feel free to add your own stuff.
    """
    s = s.lower()
    # Replace ips
    # Isolate punctuation
    s = re.sub(r'([\'\"\.\(\)\!\?\-\\\/\,])', r' \1 ', s)
    # Remove some special characters
    s = re.sub(r'([\;\:\|•«\n])', ' ', s)
    # Replace numbers and symbols with language
    s = s.replace('&', ' and ')
    s = s.replace('@', ' at ')
    s = s.replace('0', ' zero ')
    s = s.replace('1', ' one ')
    s = s.replace('2', ' two ')
    s = s.replace('3', ' three ')
    s = s.replace('4', ' four ')
    s = s.replace('5', ' five ')
    s = s.replace('6', ' six ')
    s = s.replace('7', ' seven ')
    s = s.replace('8', ' eight ')
    s = s.replace('9', ' nine ')
    return s


def fe(doc):
    s = doc
    s = normalize(s)
    s = s.split(' ')
    s = s[0:max_length]
    s = [a.strip() for a in s]
    s = [a for a in s if a in ft]
    x = np.zeros((max_length, vec_size))
    x[0:len(s)] = np.array([ft[a] for a in s])
    return x

def avg_auc(clf, X, y):
    aucs = []
    y_pred = clf.predict_proba(X)
    for i in range(y.shape[1]):
        aucs.append(roc_auc_score(y[:, i], y_pred[i][:, 1]))
    return np.mean(aucs)


class Classifier:

    def __init__(self):
        pass

    def fit(self, X, y):
        inp = Input(shape=(max_length, vec_size))
        x = GRU({nb_units})(inp)
        out = Dense(nb_outputs, activation='sigmoid')(x)
        model = Model(inputs=inp, outputs=out)
        opt = Adam(lr={lr})
        model.compile(loss='binary_crossentropy', optimizer=opt)
        self.model = model
        def gen():
            while True:
                for i in range(0, len(X), batch_size):
                    xb = X[i:i+batch_size]
                    yb = y[i:i+batch_size]
                    xb = [fe(s) for s in xb]
                    xb = np.array(xb)
                    yield xb, yb
                #auc = avg_auc(self, X, y)
                #print('Train auc : ' + str(auc))

        steps_per_epoch = len(X) // batch_size
        model.fit_generator(gen(), steps_per_epoch=steps_per_epoch, epochs={epochs})
        id_ = str(uuid.uuid4())
        filename = '.cache/gru' + id_ + '.h5'
        print('Saving the model into ' + filename)
        self.model.save(filename)
    
    def predict(self, X):
        pr = self.predict_proba(X)
        pr = np.array(pr)#6,ex,2
        pr = pr.transpose((1, 0, 2))
        pr = pr[:, :, 1]
        return (pr > 0.5).astype('float32')

    def predict_proba(self, X):
        yl = []
        for i in range(0, len(X), batch_size):
            xb = X[i:i + batch_size]
            xb = [fe(x) for x in xb]
            xb = np.array(xb)
            y = self.model.predict(xb)
            yl.append(y)
        y = np.concatenate(yl, axis=0)
        out = []
        for i in range(y.shape[1]):
            o = np.vstack((1 - y[:, i], y[:, i])).T
            out.append(o)
        return out
