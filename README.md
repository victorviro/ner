

```
cd src
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

```
python -m spacy download es_core_news_sm
```

Download the needed data
```
wget https://bitbucket.org/delectateam/nlptrainingexam/raw/aa9ea86fa4795ef2bcba2af622add9a8e69c6621/resources/ner/KnowledgeBase_subset.xlsx -P data/raw


wget https://bitbucket.org/delectateam/nlptrainingexam/raw/aa9ea86fa4795ef2bcba2af622add9a8e69c6621/resources/ner/mods%20%2Badvs.xlsx -P data/raw


wget https://bitbucket.org/delectateam/nlptrainingexam/raw/aa9ea86fa4795ef2bcba2af622add9a8e69c6621/resources/ner/ner_corpus.json -P data/raw

```