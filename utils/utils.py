def get_persona(file_name: str, file_extension: str = '.txt'):
    with open(f'prompts/oss-20b-synthetic-persona/{file_name}{file_extension}', 'r') as file:
        persona = file.read()
    return persona
