from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
import json
from style_extract import tokenizer
import collections


# Model reconstruction from JSON file
with open('model_architecture.json', 'r') as f:
    model = model_from_json(f.read())
with open('model_params.json', 'r') as f:
    data = json.load(f)
    word2ind = collections.defaultdict(lambda : 1, data["word2ind"])
    ind2word = {i:l for l,i in word2ind.items()}
    label2ind = data["label2ind"]
    ind2label = {i:l for l,i in label2ind.items()}
    ind2label[0] = 'n'
    maxlen = data["maxsize"]
    
    
def autotag(text):
    X = [tokenizer(line.strip(), lower=False, enum=False, numeric=False) for line in text.split('\n')]
    X_enc = [[word2ind[tokenizer(c, split=False, enum=True, numeric=True)] for c in x] for x in X]
    X_enc = pad_sequences(X_enc, maxlen=maxlen)
    y_enc = model.predict(X_enc).argmax(2)
    y=[ind2label[l] for row in y_enc for l in row]
    lines = []
    for row in zip(X,y_enc):
        lines.append([])
        for word, label in zip(reversed(row[0]), reversed(row[1])):
            tag = ind2label[label]
            lines[-1].insert(0, f"<{tag}>{word}</{tag}>" if tag!='n' else word)
    html="<br>".join([' '.join(line) for line in lines])
    for tag in ind2label.values():
        html = html.replace(f"</{tag}> <{tag}>", " ")
    return html
