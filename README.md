# Named-Entity-Recognition Workshop

In this workshop, we would learn how to automatically style ( **bold** , 	*Italics*, etc. ) a word according to context.

We learn styling from html files automatically and apply them to raw text.

This project is used mainly to demonstrate deep-learning implementation of named-entity-recognition (NER) models.

## Preparing a Google Colab environment
##### Recommended: training is about x10 faster than a local environment
1. Google Colab notebooks (and other resources) are located in Google Drive under `Colab Notebooks` directory.  
If you are using Colab for the first time, open [Colab](https://colab.research.google.com/notebook) and save one of the example notebooks. The notebook will be saved to `Colab Notebooks` directory.
1. Upload folder (repo content + data zip file) to your Google Drive. Make sure `nlp_ner_workshop` folder is located in your `Colab Notebooks` folder.  
Due to Google Drive quota issues make sure **not** to unzip the data file.
1. Open one of the example notebooks, change the `GOOGLE_COLAB` to `True`, and run all to test it.
1. You might need to configure your `Runtime type` to `Python 3` and set the `Hardware accelerator` to `GPU`. Both located in Runtime=>Change runtime type.

## Preparing a local environment
##### Note: in case you are not using Colab
1. Make sure Python3 is installed.
2. You can create a virtual environment (recommended) using `python3 -m virtualenv ner_ws`
3. To activate your virtual env, run: `source ner_ws/bin/activate`
4. Now install all of the requirements: `pip3 install -r requirements.txt`[](https://www.python.org/downloads/release/python-364/)

## Training a model
  1. Download data from [our Google drive](https://drive.google.com/open?id=1pWP7nPeopBe9Qf-E1xheN9TLKvITulBx)
  1. Save the `.zip` file in the `data/` folder.
  1. Run `style_learn.py` to train an NER model.
  1. Run `server.py` to evaluate your model in the browser.


For more details, contact me at [goren.ml](http://www.goren.ml) .
