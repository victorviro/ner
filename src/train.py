import random
from pathlib import Path

from tqdm import tqdm
import spacy
from spacy.util import minibatch, compounding
from sklearn.model_selection import train_test_split

from utils import get_json_from_file_path
from constants import (PREPROCESSED_DATA_PATH, ITERATIONS_NUMBER, 
                       OUTPUT_MODEL_PATH)
from evaluate import evaluate


def train():

    data_path = f'{PREPROCESSED_DATA_PATH}/training_data.json'
    data = get_json_from_file_path(data_path)[:1000]
    train_data, test_data = train_test_split(data, train_size=0.7, 
                                             random_state=4)
    num_training_examples = len(train_data)

    # Create an empty Spanish model
    nlp = spacy.blank('es')  
    print("Created blank 'en' model")
    # Create the pipeline ner component and add them to the pipeline
    ner = nlp.create_pipe('ner')
    nlp.add_pipe(ner, last=True)

    # Add labels
    for _, annotations in train_data:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    
    # Train NER
    optimizer = nlp.begin_training()
    for itn in range(ITERATIONS_NUMBER):
        random.shuffle(train_data)
        losses = {}
        # Batch up the examples using spaCy's minibatch
        batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))

        with tqdm(total=num_training_examples, leave=False) as pbar:
            # for text, annotations in tqdm(train_data):
            for batch in tqdm(batches):
                texts, annotations = zip(*batch)
                # Update the model: At each word, it makes a prediction, consults the 
                # annotations. If it was wrong, it adjusts its weights so that the
                # correct action will score higher next time.
                nlp.update(
                    texts,  
                    annotations,  
                    drop=0.5,  
                    sgd=optimizer,
                    losses=losses)
                pbar.update(len(texts))
            
            # Scoreing in training and test data
            training_scores = evaluate(nlp, train_data)
            test_scores = evaluate(nlp, test_data)
            training_f_score = training_scores.get("ents_f")
            test_f_score = test_scores.get("ents_f")
            print(f'Loss: {losses.get("ner")}')
            print(f'Training F-score: {training_f_score}; Test F-score: {test_f_score}')


    # Save the model to output directory
    if OUTPUT_MODEL_PATH is not None:
        output_model_path = Path(OUTPUT_MODEL_PATH)
        # Create the directory if it does not exist
        if not output_model_path.exists():
            output_model_path.mkdir()
        nlp.to_disk(output_model_path)
        print(f'Saved model to {output_model_path}')
