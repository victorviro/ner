# Train a NER model

## Description
In this repository, we train a **Name Entity Recognition (NER)** model with new entities in a corpus of sentences from restaurant reviews. 

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
Preprocess the data (in the proper format to train the NER model). The data processed will be stored in the directory `data/processed`.
```shell
python src/preprocess_data.py
```

## Training
Train the NER model for 5 epochs (the number of epochs can be modified in the file `constants.py`). The model will be stored in the directory `model/ner`.
```shell
python src/train.py
```

## Evaluation
Evaluate the model in the test data computing some metrics of the model. A description of the metrics used as well as an introduction to NER models is available [here](https://gist.github.com/victorviro/a70622009864e55a27c4f6d85413ea89).
```shell
python src/evaluate.py
```

## Test
Show model predicions of some random reviews in the test data. The number of predicitions to show can be modified in the file `constants.py`.

```shell
python src/test.py
```
