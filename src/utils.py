import json

import pandas as pd

# Json utils
def get_json_from_file_path(file_path):

    with open(file_path) as f:
        data = json.load(f)

    return data

def save_json_file(file_path, content_dict):

    try:
        with open(file_path, 'w') as output_file:
            json.dump(content_dict, output_file, default=str)

    except Exception as e:
        print(f'Error trying to save the output json. Message: {e}')



# DataFrame utils
def fill_null_rows_with_previous_value(df, col_names):
    # Try to do: return df.fillna(method='ffill', subset=col_names)
    df[col_names] = df[col_names].fillna(method='ffill')
    return df

def remove_rows_with_null(df, col_names):
    return df.dropna(axis=0, subset=col_names)

def group_columns_by_row(df, col_name, rows_col_to_group):
    # Try to only use only col_name, group all cols basing in col_name
    return df.groupby(col_name)[rows_col_to_group].apply(list)