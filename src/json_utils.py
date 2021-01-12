import json

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



