
import spacy 
from spacy.gold import GoldParse
from spacy.scorer import Scorer
from sklearn.model_selection import train_test_split

from utils import get_json_from_file_path
from constants import PROCESSED_DATA_PATH, OUTPUT_MODEL_PATH


def evaluate_model():
    """ 
    Load the NER model and evaluate it in the test data.
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
    print('Computing metrics...')
    scores = evaluate(ner_model, test_data)
    # General metrics of the model
    f_score = scores.get('ents_f')
    precision = scores.get('ents_p')
    recall = scores.get('ents_r')
    print('\nScoring:')
    print(f'F-score: {f_score}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')

    # Get the specific scores for each entity 
    scores_per_entity = scores.get('ents_per_type')
    # Get the F-score of the entities
    f_scores_of_entities = []
    for entity_scores in scores_per_entity.values():
        f_scores_of_entities.append(entity_scores['f'])
    # Compute the macro averaged F-score
    macro_avg_f_score = sum(f_scores_of_entities)/len(f_scores_of_entities)
    print(f'Macro averaged F-score: {macro_avg_f_score}')
    
    print('\nScores per entity;')
    print('{:<15} {:<10} {:<10} {:<10}'.format('Entity','F-score','Precision','Recall'))
    for key, value in scores_per_entity.items():
        entity = key
        f, p, r = value['f'], value['p'], value['r']
        print('{:<15} {:<10.2f} {:<10.2f} {:<10.2f}'.format(entity, f, p, r))


def evaluate(ner_model, examples):
    """ 
    Evaluate the NER model on the examples given.
    """
    # The Scorer computes and stores evaluation scores
    scorer = Scorer()
    for text, annotations in examples:
        # Process the text to get entities predicted
        document = ner_model.make_doc(text)
        correct_annotations = GoldParse(document, entities=annotations['entities'])
        predicted_annotations = ner_model(text)
        # Update the evaluation scores from the document
        scorer.score(predicted_annotations, correct_annotations)
    return scorer.scores

if __name__ == '__main__':
    evaluate_model()
