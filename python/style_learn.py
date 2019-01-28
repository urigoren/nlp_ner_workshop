import sys
import collections
import json
from zipfile import ZipFile
import numpy as np
from keras.layers import TimeDistributed, Dense, Activation, Input
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split


def read_conll_file(in_file, maxsize=256,):
    all_x = []
    seq = []
    with open(in_file, 'r', errors='ignore') as f:
        for line in f.readlines():
            try:
                x, y = line.strip().split(' ', 1)
                seq.append((x, y))
            except:
                if any(seq):
                    all_x.append(seq)
                    seq = []
        if any(seq):
            all_x.append(seq)
    lengths = [len(x) for x in all_x]
    print('Input sequence length range: ', max(lengths), min(lengths))
    short_x = [x for x in all_x if len(x) <= maxsize]
    print('# of short sequences: {n}/{m} '.format(n=len(short_x), m=len(all_x)))
    X = [[c[0] for c in x] for x in short_x]
    y = [[c[1] for c in y] for y in short_x]
    return X, y


def read_json_zip_file(in_file, maxsize=256, read_limit=1000):
    with ZipFile(in_file) as z:
        all_x = []
        i = 0
        for fname in z.filelist:
            with z.open(fname) as f:
                all_x += json.load(f)
                i += 1
            if i >= read_limit:
                break
    lengths = [len(x) for x in all_x]
    print('Input sequence length range: ', max(lengths), min(lengths))
    short_x = [x for x in all_x if len(x) <= maxsize]
    print('# of short sequences: {n}/{m} '.format(n=len(short_x), m=len(all_x)))
    X = [[c[0] for c in x] for x in short_x]
    y = [[c[1] for c in y] for y in short_x]
    return X, y


def encode(x, n):
    """One-hot encoding"""
    result = np.zeros(n)
    result[x] = 1
    return result


def score(yh, pr):
    coords = [np.where(yhh > 0)[0][0] for yhh in yh]
    yh = [yhh[co:] for yhh, co in zip(yh, coords)]
    ypr = [prr[co:] for prr, co in zip(pr, coords)]
    fyh = [c for row in yh for c in row]
    fpr = [c for row in ypr for c in row]
    return fyh, fpr


def build_model(max_sentence_length, vocab_size, num_tags):
    """ Compiles a keras model """
    l_input = Input(shape=(max_sentence_length,))
    l_embed = Embedding(vocab_size, embedding_size, input_length=max_sentence_length, mask_zero=True)(l_input)
    l_lstm = LSTM(hidden_size, return_sequences=True)(l_embed)
    l_dense = TimeDistributed(Dense(num_tags))(l_lstm)
    l_active = Activation('softmax')(l_dense)
    model = Model(inputs=l_input, outputs=l_active)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


def main():
    if in_file.lower().endswith('.txt'):
        print("Reading conll format")
        X, y = read_conll_file(in_file, maxsize)
    elif in_file.lower().endswith('.zip'):
        X, y = read_json_zip_file(in_file, maxsize)
    else:
        raise SystemError("unknown input file extension")

    corpus = (c for x in X for c in x)

    ind2word = ["{pad}", "{unk}"] + [w for w, c in collections.Counter(corpus).items() if c >= min_word_freq]
    word2ind = collections.defaultdict(lambda: 1, {word: index for index, word in enumerate(ind2word)})
    ind2label = ["{pad}"] + list(set([c for x in y for c in x]))
    label2ind = {label: index for index, label in enumerate(ind2label)}
    with open('model_params.json', 'w') as f:
        json.dump({
            "word2ind": dict(word2ind),
            "label2ind": dict(label2ind),
            "maxsize": maxsize
        }, f)
    print('Vocabulary size:', len(word2ind), len(label2ind))

    maxlen = max([len(x) for x in X])
    print('Maximum sequence length:', maxlen)

    X_enc = [[word2ind[c] for c in x] for x in X]
    max_label = len(label2ind)
    y_enc = [[0] * (maxlen - len(ey)) + [label2ind[c] for c in ey] for ey in y]
    y_enc = [[encode(c, max_label) for c in ey] for ey in y_enc]

    X_enc = pad_sequences(X_enc, maxlen=maxlen)
    y_enc = pad_sequences(y_enc, maxlen=maxlen)

    X_train, X_test, y_train, y_test = train_test_split(X_enc, y_enc, test_size=11 * 32, train_size=45 * 32,
                                                        random_state=random_state)
    print('Training and testing tensor shapes:', X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    max_features = len(word2ind)
    out_size = len(label2ind)

    model = build_model(maxlen, max_features, out_size)

    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))
    score = model.evaluate(X_test, y_test, batch_size=batch_size)
    print('Raw test score:', score)

    pr = model.predict(X_train).argmax(2)
    yh = y_train.argmax(2)
    fyh, fpr = score(yh, pr)
    print('Training accuracy:', accuracy_score(fyh, fpr))
    print('Training confusion matrix:')
    print(confusion_matrix(fyh, fpr))
    precision_recall_fscore_support(fyh, fpr)

    pr = model.predict(X_test).argmax(2)
    yh = y_test.argmax(2)
    fyh, fpr = score(yh, pr)
    print('Testing accuracy:', accuracy_score(fyh, fpr))
    print('Testing confusion matrix:')
    print(confusion_matrix(fyh, fpr))
    precision_recall_fscore_support(fyh, fpr)

    # Save the model architecture
    with open('model_architecture.json', 'w') as f:
        f.write(model.to_json())

    model.save_weights('model_weights.h5')


if __name__ == "__main__":
    in_file = sys.argv[1]
    min_word_freq = 2
    batch_size = 32
    epochs = 40
    embedding_size = 128
    hidden_size = 32
    maxsize = 256
    random_state = 1442
    main()
