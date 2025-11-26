import json
def load_json(file_path):
    """Reads JSON data from a file and returns the parsed JSON object."""
    with open(file_path, 'r') as file:
        json_string = file.read()
    return json.loads(json_string)