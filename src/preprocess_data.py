
import pandas as pd

from json_utils import get_json_from_file_path, save_json_file
from constants import (CONCEPTS_FILE_PATH, MODIFIERS_FILE_PATH, 
                       CORPUS_FILE_PATH, PREPROCESSED_DATA_PATH)
from df_utils import (fill_null_rows_with_previous_value, remove_rows_with_null,
                      remove_rows_overlapping_names, group_columns_by_row)
from spacy.matcher import PhraseMatcher
import es_core_news_sm


def process_data():

    # Concepts data
    # Load dataset in pandas DataFrame
    concepts_df = pd.read_excel(CONCEPTS_FILE_PATH, header=0)
    #concepts_df = pd.read_html(CONCEPTS_FILE_PATH)
    # Fill null values of column "Concept" with the previous value
    filled_df = fill_null_rows_with_previous_value(concepts_df, ['Concept'])
    # Remove rows with a null value in the column "Name"
    cleaned_df = remove_rows_with_null(filled_df, ['Name'])
    # Remove rows with an overlapping name
    cleaned_df = remove_rows_overlapping_names(cleaned_df, 'Name')
    # Group the raw values of the column "Name" by the column "Concept"
    grouped_df = group_columns_by_row(cleaned_df, 'Concept', 'Name')
    # Convert the DataFrame to a dictionary
    concept_and_names = grouped_df.to_dict()

    # Modifiers data
    # Load dataset in pandas DataFrame
    modifiers_df = pd.read_excel(MODIFIERS_FILE_PATH, header=0)
    # Get lists with adjectives and advebrs 
    adjectives = set(modifiers_df['ADJETIVOS'].to_list())
    adverbs = set(modifiers_df['ADVERBIOS'].dropna().to_list())
    modifiers = get_modifiers(adjectives, adverbs)
    modifiers_and_terms = {"modifier": modifiers}

    label_and_terms = dict(concept_and_names, **modifiers_and_terms)

    reviews = get_json_from_file_path(CORPUS_FILE_PATH)

    data = get_train_data(reviews, label_and_terms)

    training_data_path = f'{PREPROCESSED_DATA_PATH}/training_data.json'
    save_json_file(training_data_path, data)


# if __name__ == "__main__":
#     process_data





# Get the list with the final modifiers mixed
def get_modifiers(adjectives, adverbs):
    terms = []
    for adjective in adjectives:
        for adverb in adverbs:
            term = f'{adverb} {adjective}' 
            terms.append(term)

    return terms

def get_matches_in_train_format(text, label_and_terms, nlp):
    """
    Match entities in a text and return them in the format for train data.
    We use PhraseMatcher to find words or phrases in texts based on patterns 
    and rules
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
    Generate the training data for the NER model by matching terms
    """
    nlp = es_core_news_sm.load()
    train_data = []
    count = 0
    for review in reviews:

        matches_in_text = get_matches_in_train_format(review, label_and_terms, nlp)

        matches_info_in_text = {"entities": matches_in_text}
        review_row = (review, matches_info_in_text)
        train_data.append(review_row)
        if count%100==0:
            print(f'Analyzing review number: {count}')
        count += 1
    return train_data