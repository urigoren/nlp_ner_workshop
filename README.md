# Named-Entity-Recognition Workshop

In this workshop, we would learn how to automatically style ( **bold** , 	*Italics*, etc. ) a word according to context.

We learn styling from html files automatically and apply them to raw text.

This project is used mainly to demonstrate deep-learning implementation of named-entity-recognition (NER) models.
## Preparing the environment (locally)
 #####Note: in case you are not using Colab
1. Make sure Python3 is installed.
2. You can create you virtual environment (recommended) using `python3 -m virtualenv ner_ws`
3. To activate your virtual env, run: `source ner_ws/bin/activate`
4. Now install all of the requirements: `pip3 install -r requirements.txt`[](https://www.python.org/downloads/release/python-364/)
## Usage

  1. Run `style_extract.py` to generate training files from `html`.
  1. Put the `.zip` file in the `data/` folder.
  1. Run `style_learn.py` to train an NER model.
  1. Run `server.py` to evaluate your model in the browser.


For more details, contact me at [goren.ml](http://www.goren.ml) .
