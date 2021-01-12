
import pandas as pd



def fill_null_rows_with_previous_value(df, col_names):
    # Try to do: return df.fillna(method='ffill', subset=col_names)
    df[col_names] = df[col_names].fillna(method='ffill')
    return df

def remove_rows_with_null(df, col_names):
    return df.dropna(axis=0, subset=col_names)

def remove_rows_overlapping_names(df, col_name):
    return df[df[col_name].str.split().str.len().lt(2)]

def group_columns_by_row(df, col_name, rows_col_to_group):
    # Try to only use only col_name, group all cols basing in col_name
    return df.groupby(col_name)[rows_col_to_group].apply(list)