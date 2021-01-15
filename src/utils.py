import json


# Json utils
def get_json_from_file_path(file_path):
    
    try:
        with open(file_path) as f:
            data = json.load(f)

    except Exception as e:
        print(f'Error trying to load the json file.\nMessage: {e}')

    return data

def save_json_file(file_path, content_dict):

    try:
        with open(file_path, 'w') as output_file:
            json.dump(content_dict, output_file, default=str)

    except Exception as e:
        print(f'Error trying to save the output json.\nMessage: {e}')



# DataFrame utils
def fill_null_rows_with_previous_value(df, col_names):
    """ 
    Fill null values of the specified columns by the non-null 
    previous values of that columns.
    """
    df[col_names] = df[col_names].fillna(method='ffill')
    return df

def remove_rows_with_null(df, col_names):
    """ 
    Remove rows with null values on the specified columns.
    """
    return df.dropna(axis=0, subset=col_names)

def group_columns_by_row(df, col_name, col_to_group):
    """ 
    Group the row values of a column in a list by the values 
    of another column. 
    """
    return df.groupby(col_name)[col_to_group].apply(list)
