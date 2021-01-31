# Train a NER model

## Description
In this repository, we train a **Name Entity Recognition (NER)** model with new entities in a corpus of sentences from restaurant reviews. We use the python library [spaCy](https://spacy.io/). A description of the NER task, and a summary of different approaches to train sequence labeling tasks, is available [here](https://nbviewer.jupyter.org/github/victorviro/Deep_learning_python/blob/master/Deep_learning_methods_for_sequence_labeling_NLP.ipynb). A video explaining how the NER system from spaCy works, is available [here](https://youtu.be/sqDHBH9IjRU).

## Steps
- First, we prepare the data in the proper format to train the NER model in spaCy (by matching the search terms of the new entities in the texts of the reviews). 
- Then, we train the model. 
- Finally, we evaluate the model and show some predictions done by the model in the test data.

## Set up
Download the needed data.
```shell
wget https://bitbucket.org/delectateam/nlptrainingexam/raw/aa9ea86fa4795ef2bcba2af622add9a8e69c6621/resources/ner/KnowledgeBase_subset.xlsx -P data/raw

wget https://bitbucket.org/delectateam/nlptrainingexam/raw/aa9ea86fa4795ef2bcba2af622add9a8e69c6621/resources/ner/mods%20%2Badvs.xlsx -P data/raw

wget https://bitbucket.org/delectateam/nlptrainingexam/raw/aa9ea86fa4795ef2bcba2af622add9a8e69c6621/resources/ner/ner_corpus.json -P data/raw
```

Create virtual environment, and install requirements.
```shell
cd src
python3 -m virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data preparation
Preprocess the data in the proper format to train the NER model. The data processed will be stored in the directory `data/processed`. This process only take a couple of minutes.
```shell
python src/preprocess_data.py
```

## Training
Train the NER model for 5 epochs (the number of epochs can be modified in the file `constants.py`). The model will be stored in the directory `model/ner`. This process takes a few minutes.
```shell
python src/train.py
```

## Evaluation
Evaluate the model in the test data computing some metrics of the model.
```shell
python src/evaluate.py
```

Training the model for 5 epochs, it gets a F1-score of `0.99`, and a macro-averaged F1-score of `0.99`.

## Test
Show model predictions of some random reviews in the test data. The number of predictions to show can be modified in the file `constants.py`.

```shell
python src/test.py
```
