def tfidf():
    code = open('models/tfidf.py', 'r').read()
    return {
        'codes':{
            'classifier': code,
        },
        'info':{
        }
    }
