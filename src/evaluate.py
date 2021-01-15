import random

import spacy 
from spacy.gold import GoldParse
from spacy.scorer import Scorer
from sklearn.model_selection import train_test_split

from utils import get_json_from_file_path
from constants import (PROCESSED_DATA_PATH, OUTPUT_MODEL_PATH)


def evaluate_and_test_model():
    """ 
    Evaluate and test the NER model in the test data.
    """

    # Get the processed data (in proper format to evaluate the NER model)
    data = get_json_from_file_path(PROCESSED_DATA_PATH)
    # Split the dataset for training and test as we did for training
    train_data, test_data = train_test_split(data, train_size=0.7, 
                                             random_state=4)

    # Load the model trained
    try:
        ner_model = spacy.load(OUTPUT_MODEL_PATH)
    except Exception as err:
        msg = f'Could not load the model. Error: {err}'
        raise Exception(msg)

    # Compute evaluation scores
    scores = evaluate(ner_model, test_data)
    # General metrics of the model
    F_score = scores.get('ents_f')
    precision = scores.get('ents_p')
    recall = scores.get('ents_r')
    print('\nScoring:')
    print(f'F-score: {F_score}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')

    # Get the specific scores for each entity 
    scores_per_entity = scores.get('ents_per_type')
    entity_f_scores = [ent_scores['f'] for ent_scores in scores_per_entity.values()]
    # Compute the macro averaged F-score
    m_a_f_score = sum(entity_f_scores)/len(entity_f_scores)
    print(f'Macro averaged F-score: {m_a_f_score}')
    
    print('\nScores per entity;')
    print('{:<15} {:<10} {:<10} {:<10}'.format('Entity','F-score','Precision','Recall'))
    for key, value in scores_per_entity.items():
        entity = key
        f, p, r = value['f'], value['p'], value['r']
        print('{:<15} {:<10.2f} {:<10.2f} {:<10.2f}'.format(entity, f, p, r))
    

    # Show some predictions done by the model
    print('\nShow some predictions in the test dataset:')
    for i in random.choices(range(len(test_data)), k=4): 
        text, annotations = test_data[i]
        print(f'\nText of the review: {text}')
        print(f'Ground truth entities: {annotations["entities"]}')
        doc = ner_model(text)
        entities_predicted = [(ent.start_char, ent.end_char, ent.label_, ent.text) 
                              for ent in doc.ents]
        print(f'Entities predicted by the model: {entities_predicted}')


def evaluate(ner_model, examples):
    """ 
    Evaluate the NER model on the examples given.
    """
    # The Scorer computes and stores evaluation scores
    scorer = Scorer()
    for input_, annot in examples:
        # Process the text to get entities predicted
        doc = ner_model.make_doc(input_)
        
        gold = GoldParse(doc, entities=annot['entities'])
        pred_value = ner_model(input_)
        # Update the evaluation scores from the doc
        scorer.score(pred_value, gold)
    return scorer.scores

if __name__ == '__main__':
    evaluate_and_test_model()
