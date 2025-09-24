import configparser
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from . import common
from . import db

# Module : ia.py
# Ce module définit la classe 'ia', qui est utilisée pour rassembler des données, entraîner un modèle de RandomForest
# sur des séries temporelles et réaliser des prédictions futures en utilisant un fenêtrage glissant.

class predictopi:
    """Classe de prédiction de séries temporelles financières avec un modèle RandomForest.

    Cette classe permet de:
      - Charger la configuration et instancier une connexion à la base de données.
      - Rassembler les données de la table 'prices', en séparant les valeurs normales ('n') et foil ('f') pour chaque produit.
      - Entraîner un modèle RandomForest sur des fenêtres temporelles extraites des séries, avec normalisation via StandardScaler.
      - Effectuer des prédictions en chaîne pour prévoir les prochaines valeurs d'une série.

    Attributs principaux:
      temp_folder, db, window_size, nb_predictions, rf_model, scaler_X, series_dict.
    """
    def __init__(self, reset = False):
        """Initialise l'objet ia en chargeant la configuration, en établissant la connexion à la base de données, et en préparant le modèle.

        Si reset est False et que les modèles sauvegardés existent, ils sont chargés ; sinon, de nouveaux objets sont créés.
        """
        # Lecture du fichier de configuration
        config = configparser.ConfigParser()
        config.read('config.ini')

        # Définition du dossier temporaire pour stocker les modèles
        self.temp_folder = config.get('Folders', 'temp', fallback='./data/')
        # Création d'une instance de connexion à la base de données
        self.db = db.dbMkmPy()

        # Paramètres de la fenêtre pour les séries temporelles et le nombre de prédictions
        self.window_size = 4
        self.nb_predictions = 7

        # Chargement du modèle et du scaler depuis le dossier temporaire si disponibles,
        # sinon, initialisation d'un nouveau scaler et modèle RandomForest
        if( reset == False and 
           os.path.exists(self.temp_folder+"rf_model_mem.joblib") and 
           os.path.exists(self.temp_folder+"scaler_X.joblib") ):
            self.rf_model = joblib.load(self.temp_folder+"rf_model_mem.joblib")
            self.scaler_X = joblib.load(self.temp_folder+"scaler_X.joblib")
        else:
            self.scaler_X = StandardScaler()
            self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

    def gatherData(self, limit=None):
        """Collecte les données depuis la base de données.

        Récupère la date maximale présente dans 'prices' et, selon le paramètre limit, sélectionne les données correspondantes.
        Construit un dictionnaire dans lequel chaque clé est l'ID d'un produit et la valeur est un sous-dictionnaire contenant:
          - 'n' : série des valeurs normales
          - 'f' : série des valeurs foil

        Retourne ce dictionnaire de séries.
        """
        # Récupération de la date maximale présente dans les données
        self.maxDate = self.db.get1value("SELECT MAX(date_data) FROM prices")
        if( limit is not None ):
            results = self.db.query(f"SELECT idProduct, avg1, avg1_foil FROM prices WHERE date_data >= DATE_SUB('{self.maxDate}', INTERVAL {limit} DAY) ORDER BY idProduct, date_data")
        else:
            results = self.db.query("SELECT idProduct, avg1, avg1_foil FROM prices ORDER BY idProduct, date_data")

        # Optimisation : pré-grouper les données pour éviter np.append
        temp_dict = {}
        for row in results:
            product_id = row['idProduct']
            if product_id not in temp_dict:
                temp_dict[product_id] = {'avg1': [], 'avg1_foil': []}
            
            temp_dict[product_id]['avg1'].append(row['avg1'] if row['avg1'] is not None else 0)
            temp_dict[product_id]['avg1_foil'].append(row['avg1_foil'] if row['avg1_foil'] is not None else 0)
        
        # Convertir en numpy arrays une seule fois
        self.series_dict = {}
        for product_id, data in temp_dict.items():
            self.series_dict[product_id] = {
                'avg1': np.array(data['avg1'], dtype=np.float32),
                'avg1_foil': np.array(data['avg1_foil'], dtype=np.float32)
            }
        
        # Nettoyage explicite
        del results, temp_dict
        return self.series_dict
    
    def learn(self):
        """Entraîne le modèle RandomForest sur les données rassemblées.

        Les séries sont transformées en un dataset global en utilisant une fenêtre glissante, les features sont normalisées, puis le modèle est entraîné.
        Le modèle et le scaler sont ensuite sauvegardés pour une utilisation future.

        Retourne True en cas de succès.
        """

        if( self.series_dict is None ):
            self.gatherData()

        # Optimisation : calculer la taille totale nécessaire pour pré-allouer
        total_samples = 0
        for serie in self.series_dict.values():
            for serie_values in serie.values():
                if len(serie_values) > self.window_size:
                    total_samples += len(serie_values) - self.window_size
        
        if total_samples == 0:
            return False
            
        # Pré-allocation des arrays pour éviter les copies
        X_all = np.empty((total_samples, self.window_size), dtype=np.float32)
        y_all = np.empty(total_samples, dtype=np.float32)
        
        # Remplissage efficace des données
        idx = 0
        for serie in self.series_dict.values():
            for serie_values in serie.values():
                serie_len = len(serie_values)
                if serie_len > self.window_size:
                    # Utilisation de numpy slicing pour efficacité
                    for i in range(self.window_size, serie_len):
                        X_all[idx] = serie_values[i-self.window_size:i]
                        y_all[idx] = serie_values[i]
                        idx += 1

        # Normalisation des features et entraînement du modèle
        self.scaler_X.fit(X_all)
        X_all_scaled = self.scaler_X.transform(X_all)
        self.rf_model.fit(X_all_scaled, y_all)

        # Nettoyage mémoire explicite
        del X_all, X_all_scaled, y_all

        # Sauvegarde du modèle et du scaler pour réutilisation ultérieure
        joblib.dump(self.rf_model, self.temp_folder+"rf_model_mem.joblib")
        joblib.dump(self.scaler_X, self.temp_folder+"scaler_X.joblib")
        return True

    def predict(self):
        """Réalise des prédictions sur chaque série en utilisant le modèle entraîné.

        Pour chaque série et chaque type ('n' ou 'f'), les dernières valeurs sont utilisées pour générer une séquence de prédictions, qui est affichée dans la console.

        Retourne True en cas de succès.
        """
        if( self.series_dict is None ):
            self.gatherData(self.window_size)

        with open(self.temp_folder+"predicts.csv", "w", encoding="utf-8") as f_predicts:
            for product_id, series_type in self.series_dict.items():

                datacsv = {}

                for field, serie_values in series_type.items():
                    # Optimisation : travailler directement avec numpy array
                    # Utiliser une fenêtre glissante plutôt qu'une copie complète
                    if len(serie_values) < self.window_size:
                        continue
                        
                    # Initialiser avec les dernières valeurs de la série
                    current_window = serie_values[-self.window_size:].copy()
                    
                    for i in range(self.nb_predictions):
                        if i not in datacsv:
                            datacsv[i] = { 
                                'id': '0', 
                                'date_data': self.maxDate + pd.Timedelta(days=i+1), 
                                'idProduct': product_id 
                            }
                    
                        # Préparation des features - réutiliser le même array
                        features_scaled = self.scaler_X.transform(current_window.reshape(1, -1))
                        # Prédiction de la valeur suivante
                        pred = self.rf_model.predict(features_scaled)[0]
                        datacsv[i][field] = float(pred)
                        
                        # Mise à jour efficace de la fenêtre glissante
                        current_window = np.roll(current_window, -1)
                        current_window[-1] = pred
                
                #write datacsv to file
                for data in datacsv.values():
                    f_predicts.write(common.csvify(data, ['id', 'date_data', 'avg1', 'avg1_foil', 'idProduct']))

        return True

    def insertPredictionsToDB(self):
        print ("Not implemented yet")
        db.query("TRUNCATE TABLE predicts")
        db.import_csv_to_table("csvtemp/predicts.csv", self.temp_folder+"predicts", ";")