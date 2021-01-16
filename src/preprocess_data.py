
import pandas as pd
from tqdm import tqdm
from spacy.matcher import PhraseMatcher
from spacy.util import filter_spans
from spacy.tokens import Span
import es_core_news_sm

from constants import (CONCEPTS_FILE_PATH, MODIFIERS_FILE_PATH, 
                       CORPUS_FILE_PATH, PROCESSED_DATA_PATH)
from utils import (get_json_from_file_path, save_json_file,
                   fill_null_rows_with_previous_value, 
                   remove_rows_with_null, group_columns_by_row)


# Define a custom attribute on the spaCy Span object 
Span.set_extension("label", default=None)



def process_data():
    """
    Prepare the datasets with the concepts and modifiers, generate 
    and save the data in proper format to train the NER model 
    by matching the search terms of the entities in the texts
    of the reviews.
    """

    # Prepare the dataset of concepts
    # Load dataset in pandas DataFrame
    concepts_df = pd.read_excel(CONCEPTS_FILE_PATH, header=0)
    # Fill null values of column "Concept" with the previous value
    filled_df = fill_null_rows_with_previous_value(concepts_df, ['Concept'])
    # Remove rows with a null value in the column "Name"
    cleaned_df = remove_rows_with_null(filled_df, ['Name'])
    # Group the raw values of the column "Name" in a list by the column "Concept"
    grouped_df = group_columns_by_row(cleaned_df, 'Concept', 'Name')
    # Convert the DataFrame to a dictionary
    concepts_and_terms = grouped_df.to_dict()

    # Prepare the dataset of modifiers 
    # Load dataset in pandas DataFrame
    modifiers_df = pd.read_excel(MODIFIERS_FILE_PATH, header=0)
    # Get lists with adjectives and advebrs 
    adjectives = set(modifiers_df['ADJETIVOS'].to_list())
    adverbs = set(modifiers_df['ADVERBIOS'].dropna().to_list())
    # Get final list of modifiers 
    modifiers = get_modifiers(adjectives, adverbs)
    modifiers_and_terms = {"modifier": modifiers}

    # Get a dict with all entities and their search terms
    label_and_terms = dict(concepts_and_terms, **modifiers_and_terms)

    # Get the list of texts of the reviews
    reviews = get_json_from_file_path(CORPUS_FILE_PATH)
    print(f'Number of the reviews in the dataset: {len(reviews)}')

    # Get the data in proper format to train the NER model
    print('Generating the data for the NER model by matching the '
          'search terms of the entities in the texts of the reviews...')
    data = get_data(reviews, label_and_terms)
    print('Data in the proper format have been generated')

    # Save the data
    save_json_file(PROCESSED_DATA_PATH, data)
    print(f'Processed data saved in {PROCESSED_DATA_PATH}')


def get_modifiers(adjectives, adverbs):
    """ 
    Get final list of modifiers. Add adjectives and 
    adverb+adjective.
    """
    terms = []
    for adjective in adjectives:
        terms.append(adjective)
        for adverb in adverbs:
            term = f'{adverb} {adjective}' 
            terms.append(term)

    return terms

def get_matches_in_proper_format(text, label_and_terms, nlp):
    """
    Match the terms of an entity/label in a text and return them in the 
    format for the NER model. We use PhraseMatcher to find words or phrases
    in texts based on patterns.
    """

    matched_spans = [] 
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
        # Get the spans matched
        for match_id, start, end in matches:
            span = doc[start:end]
            # Update the custom attribute of the span to use it later
            span._.label = label
            matched_spans.append(span)
            
    # Remove overlaps. The (first) longest span is preferred over shorter spans
    matched_spans_filtered = filter_spans(matched_spans)
    entities = []
    for span in matched_spans_filtered:
        # Get the info of the match needed for the format of data
        match_info_in_text = (span.start_char, span.end_char, span._.label)
        entities.append(match_info_in_text)
    return entities

def get_data(reviews, label_and_terms):
    """
    Generate the data in proper format to train the NER model by 
    matching the search terms of the entities in the texts of 
    the reviews.
    """
    # Load the spaCy statistical model
    nlp = es_core_news_sm.load()
    # Disable unnneeded pipeline components
    nlp.disable_pipes('ner', 'tagger', 'parser')
    
    data = []
    for review in tqdm(reviews):
        # Match the terms of the entities in the text
        matches_in_text = get_matches_in_proper_format(review, 
                                                       label_and_terms, 
                                                       nlp)

        matches_info_in_text = {"entities": matches_in_text}
        # Create a tuple with the text and the matches
        row_data = (review, matches_info_in_text)
        data.append(row_data)

    return data


if __name__ == "__main__":
    process_data()
