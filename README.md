# Train a NER model

In this repository, we train a Name Entity Recognition (NER) model with new entities in a corpus of sentences from restaurant reviews. First, we prepare the data in the proper format to train the NER model in spaCy (by matching the search terms of the new entities in the texts of the sentences). Then, we train the model. Finally, we evaluate the model and show some predictions done by the model in the test data.


Download the needed data.
```
wget https://bitbucket.org/delectateam/nlptrainingexam/raw/aa9ea86fa4795ef2bcba2af622add9a8e69c6621/resources/ner/KnowledgeBase_subset.xlsx -P data/raw

wget https://bitbucket.org/delectateam/nlptrainingexam/raw/aa9ea86fa4795ef2bcba2af622add9a8e69c6621/resources/ner/mods%20%2Badvs.xlsx -P data/raw

wget https://bitbucket.org/delectateam/nlptrainingexam/raw/aa9ea86fa4795ef2bcba2af622add9a8e69c6621/resources/ner/ner_corpus.json -P data/raw
```

Create virtual environment via venv or conda e install requirements.
```
cd src
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```


Preprocess the data (in the proper format to train the NER model) (download a Spanish spacy model previously).
```
python -m spacy download es_core_news_sm
python src/preprocess_data.py
```

Train the model for 5 epochs (the number of epochs can be modified in the script `constants`).
```
python src/train.py
```

Evaluate the model in the test data computing some metrics of the model. A description of the metrics used as well as an introduction to NER models can be found [here](https://gist.github.com/victorviro/7109f5f0f00d3184a2fdaa2a956c7932).
```
python src/evaluate.py
```

Show model predicions of some random reviews in the test data. The number of predicitions to show can be modified in the script `constants`.

```
python src/test.py
```
