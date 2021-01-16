import random
from pathlib import Path

from tqdm import tqdm
import spacy
from spacy.util import minibatch, compounding
from sklearn.model_selection import train_test_split

from utils import get_json_from_file_path
from constants import (PROCESSED_DATA_PATH, ITERATIONS_NUMBER, 
                       OUTPUT_MODEL_PATH)
from evaluate import evaluate


def train():
    """ 
    Load the processed data, train the NER model and save it.
    """

    # Get the processed data (in proper format to train the NER model) 
    data = get_json_from_file_path(PROCESSED_DATA_PATH)
    # Split the dataset for training and test
    train_data, test_data = train_test_split(data, train_size=0.7, 
                                             random_state=4)
    num_training_examples = len(train_data)
    print(f'Number of examples in training data: {num_training_examples}')

    # Create an empty Spanish model
    nlp = spacy.blank('es')  
    print("Created empty model")
    # Create the pipeline ner component and add them to the pipeline
    ner = nlp.create_pipe('ner')
    nlp.add_pipe(ner, last=True)

    # Add labels
    for _, annotations in train_data:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])
    
    # Train the NER model
    print(f'Training the model for {ITERATIONS_NUMBER} epochs')
    optimizer = nlp.begin_training()
    for itn in range(ITERATIONS_NUMBER):
        # Shuffle the training data
        random.shuffle(train_data)
        losses = {}
        # Batch up the examples using spaCy's minibatch
        # Increase the batchsize to help the model get started
        batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))

        with tqdm(total=num_training_examples, leave=False) as pbar:
            
            for batch in tqdm(batches):
                texts, annotations = zip(*batch)
                # Update the model: It makes a prediction and if it was wrong, adjusts 
                # its weights so that the correct action will score higher next time.
                nlp.update(
                    texts,  
                    annotations,  
                    drop=0.5,  
                    sgd=optimizer,
                    losses=losses)
                pbar.update(len(texts))
            
            # Scoring in the training and test data
            training_scores = evaluate(nlp, train_data)
            test_scores = evaluate(nlp, test_data)
            training_f_score = training_scores.get("ents_f")
            test_f_score = test_scores.get("ents_f")
            print(f'Training loss: {losses.get("ner")}')
            print(f'Training F-score: {training_f_score}; Test F-score: {test_f_score}')


    # Save the model
    if OUTPUT_MODEL_PATH:
        output_model_path = Path(OUTPUT_MODEL_PATH)
        # Create the directory if it does not exist
        if not output_model_path.exists():
            output_model_path.mkdir()
        nlp.to_disk(output_model_path)
        print(f'Saved model to {output_model_path}')

if __name__ == "__main__":
    train()
