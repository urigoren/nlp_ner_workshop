import os
import json
import collections
import re
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
from style_extract import tokenizer
from tqdm import tqdm


def load_model(folder):
    """Loads model files, and returns keras model and parameters"""
    with open(folder + '/model_architecture.json', 'r') as f:
        model = model_from_json(f.read())
    model.load_weights(folder + '/model_weights.h5')
    model._make_predict_function()
    with open(folder + '/model_params.json', 'r') as f:
        data = json.load(f)
        word2ind = collections.defaultdict(lambda: 1, data["word2ind"])
        ind2word = {i: l for l, i in word2ind.items()}
        label2ind = data["label2ind"]
        ind2label = {i: l for l, i in label2ind.items()}
        ind2label[0] = 'n'
        maxlen = data["maxsize"]
    return model, {'word2ind': word2ind, 'ind2word': ind2word, 'label2ind': label2ind, 'ind2label': ind2label,
                   'maxlen': maxlen}


def predict_on_token_array(X, model, params):
    X_enc = [[params['word2ind'][x] for x in X]]
    X_enc = pad_sequences(X_enc, maxlen=params['maxlen'])
    y_enc = model.predict(X_enc).argmax(2)
    y_enc = list(y_enc)[0][-len(X):]
    return [params["ind2label"][y] for y in y_enc]


def predict_on_test_file(filename, model, params):
    ret = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            line_prediction = ' '.join(predict_on_token_array(line.split(), model, params))
            ret.append(line_prediction)
    return ret


def predict_on_test_dir(dirname, model, params):
    ret= {}
    for fname in tqdm(os.listdir(dirname)):
        if fname.endswith('.txt'):
            ret[fname] = predict_on_test_file(dirname+'/'+fname, model, params)
    return ret


def autotag(text, model, params):
    """Gets text, model and params, and outputs formatter HTML"""
    # Covert line to X_enc vector, and predict y_enc
    X = [tokenizer(line.strip(), lower=False, enum=False, numeric=False) for line in text.split('\n')]
    X_enc = [[params['word2ind'][tokenizer(c, split=False, enum=True, numeric=True)] for c in x] for x in X]
    X_enc = pad_sequences(X_enc, maxlen=params['maxlen'])
    y_enc = model.predict(X_enc).argmax(2)
    # Turn prediction to HTML
    lines = []
    for row in zip(X, y_enc):
        lines.append([])
        for word, label in zip(reversed(row[0]), reversed(row[1])):
            tag = params['ind2label'][label]
            lines[-1].insert(0, f"<{tag}>{word}</{tag}>" if tag != 'n' else word)
    html = "<br>".join([' '.join(line) for line in lines])
    # Rejoin words together, to get a cleaner view
    for tag in params['ind2label'].values():
        html = html.replace(f"</{tag}> <{tag}>", " ")
    html = re.compile('" (\w+) "').sub('"\\1"', html)
    html = re.compile(' ([\\.,:;]) ').sub('\\1 ', html)
    return html


if __name__ == "__main__":
    model, params = load_model("../model")
    X = "governing law . the parties shall obide to".split()
    y = predict_on_token_array(X, model, params)
    print(list(zip(X, y)))
