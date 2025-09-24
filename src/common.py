import datetime


def csvify(values, format):
    # Transforme un dictionnaire de valeurs en ligne CSV, avec gestion des types et des guillemets
    csvline = ""
    for key in format:
        if values[key] is None:
            csvline += 'NULL;'
        elif(isinstance(values[key], datetime.date)):
            csvline += f'{values[key].strftime("%Y-%m-%d")};'
        elif( isinstance(values[key], int) or isinstance(values[key], float) or values[key].isnumeric()  ):
            csvline += f'{values[key]};'
        else:
            csvline += f'"{str(values[key]).replace("\"", "\"\"")}";'
    return f'{csvline}\n'

