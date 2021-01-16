import os



PROJECT_PATH = os.getcwd()
CONCEPTS_FILE_PATH = f'{PROJECT_PATH}/data/raw/KnowledgeBase_subset.xlsx'
MODIFIERS_FILE_PATH = f'{PROJECT_PATH}/data/raw/mods +advs.xlsx'
CORPUS_FILE_PATH = f'{PROJECT_PATH}/data/raw/ner_corpus.json'

PROCESSED_DATA_PATH = f'{PROJECT_PATH}/data/processed/data.json'

OUTPUT_MODEL_PATH = f'{PROJECT_PATH}/models/ner'
# Number of epochs to train the model
ITERATIONS_NUMBER = 5

# Number of predictions to show when test the model
PREDICTIONS_NUMBER = 10
