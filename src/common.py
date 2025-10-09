import configparser
import datetime
import os


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

def getconfigfile():
    if os.path.exists("../config.ini"):
        config_file = "../config.ini"
    else:
        config_file = "config.ini"
        
    config = configparser.ConfigParser()

    # Read the configuration file
    config.read(config_file)
    return config  # Retourner l'objet config, pas le résultat de read()