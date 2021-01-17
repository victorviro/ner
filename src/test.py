import random

import spacy 
from sklearn.model_selection import train_test_split

from utils import get_json_from_file_path
from constants import (PROCESSED_DATA_PATH, OUTPUT_MODEL_PATH,
                       PREDICTIONS_NUMBER)

def test_model():
    """ 
    Load the NER model trained and show some predictions done by the 
    model in the test dataset.
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

    # Show some predictions done by the model
    print(f'\nShowing predictions of {PREDICTIONS_NUMBER} random reviews of the'
           ' test dataset:')
           
    for text, annotations in random.choices(test_data, k=PREDICTIONS_NUMBER):

        # Get the Doc from the text of the review
        document = ner_model(text)

        # Get the entities predicted
        entities_predicted = []
        for entity in document.ents:
            # Get the positions in the sentence, the name of the entity and the text
            entity_predicted_info = (entity.start_char, entity.end_char, 
                                     entity.label_, entity.text)
            entities_predicted.append(entity_predicted_info)

        print(f'\nText of the review: {text}')
        print(f'Ground truth entities: {annotations["entities"]}')
        print(f'Entities predicted by the model: {entities_predicted}')

if __name__ == '__main__':
    test_model()
