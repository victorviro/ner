import os



PROJECT_PATH = os.getcwd()
CONCEPTS_FILE_PATH = f'{PROJECT_PATH}/data/raw/KnowledgeBase_subset.xlsx'
MODIFIERS_FILE_PATH = f'{PROJECT_PATH}/data/raw/mods +advs.xlsx'
CORPUS_FILE_PATH = f'{PROJECT_PATH}/data/raw/ner_corpus.json'

PROCESSED_DATA_PATH = f'{PROJECT_PATH}/data/processed/training_data.json'

OUTPUT_MODEL_PATH = f'{PROJECT_PATH}/models/ner'
ITERATIONS_NUMBER = 5