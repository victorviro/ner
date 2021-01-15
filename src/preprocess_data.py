
import pandas as pd
from tqdm import tqdm
from spacy.matcher import PhraseMatcher
import es_core_news_sm

from constants import (CONCEPTS_FILE_PATH, MODIFIERS_FILE_PATH, 
                       CORPUS_FILE_PATH, PROCESSED_DATA_PATH)
from utils import (get_json_from_file_path, save_json_file,
                   fill_null_rows_with_previous_value, 
                   remove_rows_with_null, group_columns_by_row)


def process_data():
    """
    Prepare the datasets with the concepts and modifiers and generate 
    and save the training data for the NER model by matching the 
    search terms of the entities in the texts of the reviews.
    """

    # Dataset of concepts
    # Load dataset in pandas DataFrame
    concepts_df = pd.read_excel(CONCEPTS_FILE_PATH, header=0)
    # Fill null values of column "Concept" with the previous value
    filled_df = fill_null_rows_with_previous_value(concepts_df, ['Concept'])
    # Remove rows with a null value in the column "Name"
    cleaned_df = remove_rows_with_null(filled_df, ['Name'])
    # Remove rows with an overlapping name (see README for details)
    cleaned_df = cleaned_df[cleaned_df['Name'].str.split().str.len().lt(2)]
    # Group the raw values of the column "Name" in a list by the column "Concept"
    grouped_df = group_columns_by_row(cleaned_df, 'Concept', 'Name')
    # Convert the DataFrame to a dictionary
    concept_and_names = grouped_df.to_dict()

    # Dataset of modifiers 
    # Load dataset in pandas DataFrame
    modifiers_df = pd.read_excel(MODIFIERS_FILE_PATH, header=0)
    # Get lists with adjectives and advebrs 
    adjectives = set(modifiers_df['ADJETIVOS'].to_list())
    adverbs = set(modifiers_df['ADVERBIOS'].dropna().to_list())
    # Get final list of modifiers (no overlapping modifiers, see README)
    modifiers = get_modifiers(adjectives, adverbs)
    modifiers_and_terms = {"modifier": modifiers}

    # Get a dict with all entities and their search terms
    label_and_terms = dict(concept_and_names, **modifiers_and_terms)

    # Get the list of texts of the reviews
    reviews = get_json_from_file_path(CORPUS_FILE_PATH)

    # Get the training data for the NER model
    print('Generating the training data for the NER model by matching the '
          'search terms of the entities in the texts of the reviews...')
    data = get_train_data(reviews, label_and_terms)
    print('Training data generated')

    # Save the data
    save_json_file(PROCESSED_DATA_PATH, data)
    print(f'Processed data saved in {PROCESSED_DATA_PATH}')


def get_modifiers(adjectives, adverbs):
    """ 
    Get final list of modifiers of the form: adverb+adjective.
    No overlapping modifiers (see README).
    """
    terms = []
    for adjective in adjectives:
        for adverb in adverbs:
            term = f'{adverb} {adjective}' 
            terms.append(term)

    return terms

def get_matches_in_train_format(text, label_and_terms, nlp):
    """
    Match the terms of a entity/label in a text and return them in the 
    format for train data. We use PhraseMatcher to find words or phrases
    in texts based on patterns and rules.
    """

    entities = []
    for label, terms in label_and_terms.items():
        # Initialize the PhraseMatcher with the vocabulary 
        matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
        terms = set(terms)
        # Add the rule/pattern to the matcher
        patterns = [nlp.make_doc(text) for text in terms]
        matcher.add("TerminologyList", patterns)
        # Process the text
        doc = nlp(text)
        # Find all sequences matching the supplied patterns on the Doc
        matches = matcher(doc)
        # Loop over all matches
        
        for match_id, start, end in matches:
            span = doc[start:end]
            # Get the info of the match needed for the format of train data
            match_info_in_text = (span.start_char, span.end_char, label)
            entities.append(match_info_in_text)

    return entities

def get_train_data(reviews, label_and_terms):
    """
    Generate the training data for the NER model by matching the 
    search terms of the entities in the texts of the reviews.
    """
    # Load the spaCy statistical model
    nlp = es_core_news_sm.load()
    # Disable unnneeded pipeline components
    nlp.disable_pipes('ner', 'tagger', 'parser')
    train_data = []

    for review in tqdm(reviews):
        # Match the terms of a entity in the text
        matches_in_text = get_matches_in_train_format(review, label_and_terms, nlp)

        matches_info_in_text = {"entities": matches_in_text}
        # Create a tuple with the text and the matches
        review_row = (review, matches_info_in_text)
        train_data.append(review_row)

    return train_data


if __name__ == "__main__":
    process_data()
