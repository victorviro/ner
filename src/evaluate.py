import random

import spacy 
from spacy.gold import GoldParse
from spacy.scorer import Scorer

from json_utils import get_json_from_file_path
from constants import (PREPROCESSED_DATA_PATH, OUTPUT_MODEL_PATH)


def evaluate_model():

    # Geberate the test data we use for training
    data_path = f'{PREPROCESSED_DATA_PATH}/training_data.json'
    data = get_json_from_file_path(data_path)[:1000]
    random.seed(4)
    random.shuffle(data)
    examples_number = len(data)
    num_training_examples = int(examples_number*0.7)
    test_data = data[num_training_examples:]

    # Load model
    ner_model = spacy.load(OUTPUT_MODEL_PATH)

    # Compute evaluation scores
    scores = evaluate(ner_model, test_data)
    F_score = scores.get("ents_f")
    precision = scores.get("ents_p")
    recall = scores.get("ents_r")
    print('\nScoring:')
    print(f'F-score: {F_score}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')

    ents_per_type = scores.get("ents_per_type")
    entity_f_scores = [ent_type["f"] for ent_type in ents_per_type.values()]
    m_a_f_score = sum(entity_f_scores)/len(entity_f_scores)
    print(f'Macro averaged F-score: {recall}')
    
    print('\nScores per entity;')
    print("{:<15} {:<10} {:<10} {:<10}".format('Entity','F-score','Precision',"Recall"))
    for key, value in ents_per_type.items():
        entity = key
        f, p, r = value["f"], value["p"], value["r"]
        print("{:<15} {:<10.2f} {:<10.2f} {:<10.2f}".format(entity, f, p, r))
    


def evaluate(ner_model, examples):
    # The Scorer computes and stores evaluation scores.
    scorer = Scorer()
    for input_, annot in examples:
        # Process the text to get entities predicted
        doc = ner_model.make_doc(input_)
        
        gold = GoldParse(doc, entities=annot['entities'])
        pred_value = ner_model(input_)
        # Update the evaluation scores from the doc
        scorer.score(pred_value, gold)
    return scorer.scores