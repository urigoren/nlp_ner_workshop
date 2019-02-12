from urllib import request, parse
import os
import json
from sklearn.base import BaseEstimator, ClassifierMixin
from style_predict import load_model, predict_on_test_dir, predict_on_token_array, predict_on_test_file
from style_learn import fit_file


class Submission(BaseEstimator, ClassifierMixin):
    def __init__(self, submission_name, model_folder='../model', train_params={}):
        self.model_folder = model_folder
        self.submission_name = submission_name
        self.train_params = train_params

    def fit(self, X, y=None):
        assert type(X) == str
        assert os.path.exists(X)
        filename = X
        fit_file(filename, self.train_params)

    def predict(self, X, y=None):
        model, params = load_model(self.model_folder)
        if type(X)==list:
            return [predict_on_token_array(x, model, params) for x in X]
        elif os.path.exists(X):
            if os.path.isdir(X):
                return predict_on_test_dir(X, model, params)
            else:
                return predict_on_test_file(X, model, params)
        else:
            return ' '.join(predict_on_token_array(X.split(), model, params))

    def submit(self, test_folder="../data/test_txt"):
        """Submits a test data set to the goren.ml server"""
        if type(test_folder)==str:
            model, params = load_model(self.model_folder)
            test_data = predict_on_test_dir(test_folder, model, params)
        else:
            test_data = test_folder
        data = parse.urlencode({"name":self.submission_name, "submission": json.dumps(test_data)}).encode()
        req = request.Request("http://goren4u.com/nlp_ner/", data=data)
        resp = request.urlopen(req)
        return float(resp.read().decode("utf8"))


if __name__ == "__main__":
    submission = Submission("ug")
    submission.submit()
    # print(submission.submit({"a": ["n n b","b"], "b": ["n n b","b"]}))
