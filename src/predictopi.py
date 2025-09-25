
# =============================
# Module : predictopi.py
# =============================
# Ce module définit la classe 'predictopi', dédiée à la prévision du prix des cartes Magic vendues sur le site Cardmarket.
# Il permet de charger les historiques de prix, d'entraîner un modèle de machine learning (RandomForest) sur ces séries temporelles,
# puis de prédire l'évolution future des prix pour chaque carte, et d'insérer ces prévisions en base de données.

import configparser
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from . import common
from . import db

class predictopi:
    """
    Classe de prévision du prix des cartes Magic vendues sur Cardmarket, basée sur un modèle RandomForest.

    Fonctionnalités principales :
        - Chargement de la configuration et connexion à la base de données.
        - Extraction et préparation des historiques de prix (normal et foil) pour chaque carte Magic.
        - Entraînement d'un modèle RandomForest sur des fenêtres glissantes de l'historique de prix.
        - Prédiction de l'évolution future du prix pour chaque carte.
        - Export des prévisions au format CSV et insertion en base de données.

    Attributs principaux :
        - temp_folder : dossier temporaire pour les modèles et fichiers intermédiaires
        - db : instance de connexion à la base de données
        - window_size : taille de la fenêtre glissante pour l'apprentissage
        - nb_predictions : nombre de jours à prédire
        - rf_model : modèle RandomForestRegressor
        - scaler_X : normaliseur des features
        - series_dict : dictionnaire des historiques de prix par carte
    """
    def __init__(self, reset = False):
        """
        Initialise l'objet predictopi pour la prévision des prix Cardmarket :
        - Charge la configuration (chemin des fichiers, etc.)
        - Instancie la connexion à la base de données
        - Charge le modèle et le scaler s'ils existent, sinon les crée
        - Définit les paramètres de fenêtre et de prédiction

        :param reset: Si True, force la réinitialisation du modèle et du scaler
        """
        config = configparser.ConfigParser()
        config.read('config.ini')
        self.temp_folder = config.get('Folders', 'temp', fallback='./data/')
        self.db = db.dbMkmPy()
        self.window_size = 4
        self.nb_predictions = 7
        self.series_dict = None
        # Chargement ou création du modèle/scaler
        if( not reset and 
           os.path.exists(self.temp_folder+"rf_model_mem.joblib") and 
           os.path.exists(self.temp_folder+"scaler_X.joblib") ):
            self.rf_model = joblib.load(self.temp_folder+"rf_model_mem.joblib")
            self.scaler_X = joblib.load(self.temp_folder+"scaler_X.joblib")
        else:
            self.scaler_X = StandardScaler()
            # Configuration optimisée pour la prédiction de prix de cartes Magic
            self.rf_model = RandomForestRegressor(
                n_estimators=200,        # Plus d'arbres pour meilleure précision
                max_depth=15,           # Limite la profondeur pour éviter le surapprentissage
                min_samples_split=5,    # Réduit le surapprentissage sur petites séries
                min_samples_leaf=2,     # Feuilles plus générales
                max_features='sqrt',    # Utilise racine carrée du nombre de features
                n_jobs=-1,              # Utilise tous les cœurs CPU
                random_state=42,        # Reproductibilité
                oob_score=True          # Score de validation interne
            )

    def gatherData(self, limit=None):
        """
        Récupère et prépare les historiques de prix des cartes Magic depuis la base de données.
        - Récupère la date maximale disponible
        - Sélectionne les données sur la période souhaitée (limit en jours)
        - Construit un dictionnaire de séries numpy pour chaque carte (prix normal et foil)

        :param limit: Nombre de jours à récupérer (None = tout)
        :return: Dictionnaire {idCarte: {'avg1': array, 'avg1_foil': array}}
        """
        self.maxDate = self.db.get1value("SELECT MAX(date_data) FROM prices")
        if( limit is not None ):
            results = self.db.query(f"SELECT idProduct, avg1, avg1_foil FROM prices WHERE date_data >= DATE_SUB('{self.maxDate}', INTERVAL {limit} DAY) ORDER BY idProduct, date_data")
        else:
            results = self.db.query("SELECT idProduct, avg1, avg1_foil FROM prices ORDER BY idProduct, date_data")
        # Agrégation efficace des données par produit
        temp_dict = {}
        for row in results:
            product_id = row['idProduct']
            if product_id not in temp_dict:
                temp_dict[product_id] = {'avg1': [], 'avg1_foil': []}
            temp_dict[product_id]['avg1'].append(row['avg1'] if row['avg1'] is not None else 0)
            temp_dict[product_id]['avg1_foil'].append(row['avg1_foil'] if row['avg1_foil'] is not None else 0)
        # Conversion en numpy arrays
        self.series_dict = {}
        for product_id, data in temp_dict.items():
            self.series_dict[product_id] = {
                'avg1': np.array(data['avg1'], dtype=np.float32),
                'avg1_foil': np.array(data['avg1_foil'], dtype=np.float32)
            }
        del results, temp_dict
        return self.series_dict
    
    def learn(self):
        """
        Entraîne le modèle RandomForest sur les historiques de prix extraits.
        - Génère les jeux de données d'apprentissage par fenêtre glissante sur les prix des cartes
        - Normalise les features
        - Entraîne le modèle
        - Sauvegarde le modèle et le scaler sur disque

        :return: True si l'entraînement a réussi, False sinon
        """
        if( self.series_dict is None ):
            self.gatherData()
        # Calcul du nombre total d'échantillons pour pré-allocation
        total_samples = 0
        for serie in self.series_dict.values():
            for serie_values in serie.values():
                if len(serie_values) > self.window_size:
                    total_samples += len(serie_values) - self.window_size
        if total_samples == 0:
            return False
        # Pré-allocation des tableaux numpy
        X_all = np.empty((total_samples, self.window_size), dtype=np.float32)
        y_all = np.empty(total_samples, dtype=np.float32)
        # Remplissage des tableaux par fenêtre glissante
        idx = 0
        for serie in self.series_dict.values():
            for serie_values in serie.values():
                serie_len = len(serie_values)
                if serie_len > self.window_size:
                    for i in range(self.window_size, serie_len):
                        X_all[idx] = serie_values[i-self.window_size:i]
                        y_all[idx] = serie_values[i]
                        idx += 1
        # Normalisation et apprentissage
        self.scaler_X.fit(X_all)
        X_all_scaled = self.scaler_X.transform(X_all)
        self.rf_model.fit(X_all_scaled, y_all)
        # Nettoyage mémoire
        del X_all, X_all_scaled, y_all
        # Sauvegarde des objets
        joblib.dump(self.rf_model, self.temp_folder+"rf_model_mem.joblib")
        joblib.dump(self.scaler_X, self.temp_folder+"scaler_X.joblib")
        
        # Affichage du score de performance (Out-of-Bag)
        # if hasattr(self.rf_model, 'oob_score_'):
            # print(f"Score OOB du modèle : {self.rf_model.oob_score_:.4f}")
        
        return True

    def predict(self):
        """
        Prédit l'évolution future du prix de chaque carte Magic à l'aide du modèle entraîné.
        - Utilise les derniers prix connus comme point de départ
        - Prédit les prix futurs en chaîne (fenêtre glissante)
        - Écrit les prévisions dans un fichier CSV (un par carte)

        :return: True si la prédiction a réussi
        """
        if( self.series_dict is None ):
            self.gatherData(self.window_size)
        with open(self.temp_folder+"predicts.csv", "w", encoding="utf-8") as f_predicts:
            for product_id, series_type in self.series_dict.items():
                datacsv = {}
                for field, serie_values in series_type.items():
                    # Ignore les séries trop courtes
                    if len(serie_values) < self.window_size:
                        continue
                    # Démarre la prédiction avec la dernière fenêtre connue
                    current_window = serie_values[-self.window_size:].copy()
                    for i in range(self.nb_predictions):
                        if i not in datacsv:
                            datacsv[i] = { 
                                'id': '0', 
                                'date_data': self.maxDate + pd.Timedelta(days=i+1), 
                                'idProduct': product_id 
                            }
                        # Prépare les features et prédit la prochaine valeur
                        features_scaled = self.scaler_X.transform(current_window.reshape(1, -1))
                        pred = self.rf_model.predict(features_scaled)[0]
                        # Si la prédiction est négative ou nulle, on la considère comme absente
                        if( pred <= 0 ):
                            datacsv[i][field] = None
                        else:
                            datacsv[i][field] = float(pred)
                        # Met à jour la fenêtre glissante
                        current_window = np.roll(current_window, -1)
                        current_window[-1] = pred
                # Écrit les résultats dans le CSV
                for data in datacsv.values():
                    f_predicts.write(common.csvify(data, ['id', 'date_data', 'avg1', 'avg1_foil', 'idProduct']))
        return True

    def insertPredictionsToDB(self):
        """
        Insère les prévisions de prix générées dans la table prices_predict de la base de données.
        - Vide la table cible
        - Importe le fichier CSV généré dans la table
        """
        self.db.query("TRUNCATE TABLE prices_predict")
        self.db.import_csv_to_table(self.temp_folder+"predicts.csv", "prices_predict", ";")