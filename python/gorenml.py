from urllib import request, parse
import json
from pathlib import Path
from sklearn.base import BaseEstimator, ClassifierMixin
from style_predict import load_model, predict_on_test_dir, predict_on_token_array
from style_learn import fit_file
from style_extract import tokenizer


class Submitter(BaseEstimator, ClassifierMixin):
    def __init__(self, submission_name, model_folder='../model', train_params={}):
        self.model_folder = model_folder
        self.submission_name = submission_name
        self.train_params = train_params
    def fit(self, X, y=None):
        assert type(X) == str
        assert Path(X).exists()
        filename = X
        fit_file(filename, self.train_params)
    def predict(self, X, y=None):
        model, params = load_model(self.model_folder)
        return [predict_on_token_array(x, model, params) for x in X]
    def submit(self, test_folder):
        """Submits a test data set to the goren.ml server"""
        test_data = []
        for f in Path(test_folder).iterdir():
            txt = f.open("rb").read().decode("utf-8", errors="ignore")
            test_data.extend([tokenizer(t) for t in txt.split('\n')])
        try:
            test_data = predict_on_test_dir("../model/test_txt/")
            data = parse.urlencode({"name":self.submission_name, "submission": json.dumps(test_data)}).encode()
            req = request.Request("http://goren4u.com/nlp_ner/", data=data)
            resp = request.urlopen(req)
            return float(resp.read().decode("utf8"))
        except:
            return None


if __name__ == "__main__":
    # assert test_accuracy("ug", {"34048": "rsu"}) > 0
    pass