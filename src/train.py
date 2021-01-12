import random
from pathlib import Path
from tqdm import tqdm
import time

import spacy
from spacy.util import minibatch, compounding

from json_utils import get_json_from_file_path
from constants import (PREPROCESSED_DATA_PATH, ITERATIONS_NUMBER, 
                       OUTPUT_MODEL_PATH)

def train():

    data_path = f'{PREPROCESSED_DATA_PATH}/training_data.json'
    data = get_json_from_file_path(data_path)[:1000]
    random.seed(4)
    random.shuffle(data)
    examples_number = len(data)
    num_training_examples = int(examples_number*0.7)
    train_data = data[:num_training_examples]
    test_data = data[num_training_examples:]

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
            print(f'Loss: {losses.get("ner")}')


    # Save the model to output directory
    if OUTPUT_MODEL_PATH is not None:
        output_model_path = Path(OUTPUT_MODEL_PATH)
        # Create the directory if it does not exist
        # if not output_model_path.exists():
        #     output_model_path.mkdir()
        nlp.to_disk(output_model_path)
        print(f'Saved model to {output_model_path}')
