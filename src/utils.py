import json
from pandas import DataFrame


# Json utils
def get_json_from_file_path(file_path:str) -> dict:
    
    try:
        with open(file_path) as f:
            data = json.load(f)

    except Exception as e:
        print(f'Error trying to load the json file.\nMessage: {e}')

    return data

def save_json_file(file_path:str, content:dict):

    try:
        with open(file_path, 'w') as output_file:
            json.dump(content, output_file, default=str)

    except Exception as e:
        print(f'Error trying to save the output json.\nMessage: {e}')



# Pandas DataFrame utils
def fill_null_rows_with_previous_value(df:DataFrame, col_names:list) -> DataFrame:
    """ 
    Fill null values of the specified columns by the non-null 
    previous values of that columns.
    """
    df[col_names] = df[col_names].fillna(method='ffill')
    return df

def remove_rows_with_null(df:DataFrame, col_names:list) -> DataFrame:
    """ 
    Remove rows with null values on the specified columns.
    """
    return df.dropna(axis=0, subset=col_names)

def group_columns_by_row(df:DataFrame, col_name:str, 
                         col_to_group:str) -> DataFrame:
    """ 
    Group the row values of a column in a list by the values 
    of another column. 
    """
    return df.groupby(col_name)[col_to_group].apply(list)
