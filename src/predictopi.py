
# =============================
# Module : predictopi.py
# =============================
# Ce module définit la classe 'predictopi', dédiée à la prévision du prix des cartes Magic vendues sur le site Cardmarket.
# Il permet de charger les historiques de prix, d'entraîner un modèle de machine learning (RandomForest) sur ces séries temporelles,
# puis de prédire l'évolution future des prix pour chaque carte, et d'insérer ces prévisions en base de données.

import os
import joblib
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from src import common
from src import db

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
        config = common.getconfigfile()
        self.temp_folder        = config.get('Folders', 'temp', fallback='./data/')

        self.n_estimators       = config.getint('IA', 'n_estimators', fallback=200)
        self.max_depth          = config.getint('IA', 'max_depth', fallback=15)
        self.min_samples_split  = config.getint('IA', 'min_samples_split', fallback=5)
        self.min_samples_leaf   = config.getint('IA', 'min_samples_leaf', fallback=2)
        self.max_features       = config.get(   'IA', 'max_features', fallback='sqrt')
        self.n_jobs             = config.getint('IA', 'n_jobs', fallback=-1)
        self.random_state       = config.getint('IA', 'random_state', fallback=42)

        self.window_size        = config.getint('IA', 'window_size', fallback=7)
        self.nb_predictions     = config.getint('IA', 'nb_predictions', fallback=10)
        self.learn_days         = config.getint('IA', 'learn_days', fallback=None)

        self.db = db.dbMkmPy()

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
                n_estimators = self.n_estimators,	        # Plus d'arbres pour meilleure précision
                max_depth = self.max_depth,	                # Limite la profondeur pour éviter le surapprentissage
                min_samples_split = self.min_samples_split,	# Réduit le surapprentissage sur petites séries
                min_samples_leaf = self.min_samples_leaf,	# Feuilles plus générales
                max_features = self.max_features,	        # Utilise racine carrée du nombre de features
                n_jobs = self.n_jobs,	                    # Utilise tous les cœurs CPU
                random_state = self.random_state	        # Reproductibilité

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
        sql = "SELECT idProduct, avg1, avg1_foil FROM prices"
        if( limit is not None ):
            sql += f" WHERE date_data >= DATE_SUB('{self.maxDate}', INTERVAL {limit} DAY)"
        sql += " ORDER BY idProduct, date_data"
        # Lecture en flux, carte par carte (les lignes arrivent triées par idProduct) :
        # seules les séries numpy finales restent en mémoire, pas le résultat complet de la requête
        self.series_dict = {}
        current_id = None
        avg1, avg1_foil = [], []
        for product_id, price, price_foil in self.db.iterate(sql):
            if product_id != current_id:
                self._storeSerie(current_id, avg1, avg1_foil)
                current_id, avg1, avg1_foil = product_id, [], []
            avg1.append(price if price is not None else 0)
            avg1_foil.append(price_foil if price_foil is not None else 0)
        self._storeSerie(current_id, avg1, avg1_foil)
        return self.series_dict

    def _storeSerie(self, product_id, avg1, avg1_foil):
        if product_id is not None:
            self.series_dict[product_id] = {
                'avg1': np.array(avg1, dtype=np.float32),
                'avg1_foil': np.array(avg1_foil, dtype=np.float32)
            }

    def _trainingSamples(self):
        """
        Fenêtres glissantes de chaque série et prix du jour suivant.
        Les échantillons entièrement à 0 (prix absent, ex. carte sans version foil) sont écartés :
        ils n'apprennent rien et représentent une grande partie des séries.
        """
        for serie in self.series_dict.values():
            for serie_values in serie.values():
                if len(serie_values) > self.window_size:
                    windows = sliding_window_view(serie_values[:-1], self.window_size)
                    targets = serie_values[self.window_size:]
                    keep = windows.any(axis=1) | (targets != 0)
                    yield windows[keep], targets[keep]
    
    def learn(self, save=True):
        """
        Entraîne le modèle RandomForest sur les historiques de prix extraits.
        - Génère les jeux de données d'apprentissage par fenêtre glissante sur les prix des cartes
        - Normalise les features
        - Entraîne le modèle
        - Sauvegarde le modèle et le scaler sur disque

        :param save: Si False, le modèle n'est pas écrit sur disque (utile pour l'évaluation)
        :return: True si l'entraînement a réussi, False sinon
        """
        if( self.series_dict is None ):
            self.gatherData(self.learn_days)
        # Calcul du nombre total d'échantillons pour pré-allocation
        total_samples = sum(len(targets) for _, targets in self._trainingSamples())
        if total_samples == 0:
            return False
        # Pré-allocation des tableaux numpy
        X_all = np.empty((total_samples, self.window_size), dtype=np.float32)
        y_all = np.empty(total_samples, dtype=np.float32)
        # Remplissage des tableaux par fenêtre glissante
        idx = 0
        for windows, targets in self._trainingSamples():
            X_all[idx:idx+len(targets)] = windows
            y_all[idx:idx+len(targets)] = targets
            idx += len(targets)
        # Normalisation (en place, sans copie du jeu d'apprentissage) et apprentissage
        self.scaler_X.fit(X_all)
        X_all = self.scaler_X.transform(X_all, copy=False)
        self.rf_model.fit(X_all, y_all)
        # Nettoyage mémoire
        del X_all, y_all
        # Sauvegarde des objets
        if save:
            joblib.dump(self.rf_model, self.temp_folder+"rf_model_mem.joblib")
            joblib.dump(self.scaler_X, self.temp_folder+"scaler_X.joblib")
        
        # Affichage du score de performance (Out-of-Bag)
        # if hasattr(self.rf_model, 'oob_score_'):
            # print(f"Score OOB du modèle : {self.rf_model.oob_score_:.4f}")
        
        return True

    def forecast(self, windows, steps):
        """
        Prédiction récursive en batch : chaque jour prédit est réinjecté dans la fenêtre
        pour prédire le jour suivant.

        :param windows: Fenêtres de prix, forme (n, window_size)
        :param steps: Nombre de jours à prédire
        :return: Tableau numpy des prédictions, forme (n, steps)
        """
        windows = np.array(windows, dtype=np.float32)
        preds = np.empty((len(windows), steps))
        for i in range(steps):
            preds[:, i] = self.rf_model.predict(self.scaler_X.transform(windows))
            windows = np.roll(windows, -1, axis=1)
            windows[:, -1] = preds[:, i]
        return preds

    def predict(self):
        """
        Prédit l'évolution future du prix de chaque carte Magic à l'aide du modèle entraîné.
        Optimisation : prédiction en batch pour toutes les cartes et chaque type de prix, écriture du CSV en une seule fois.

        :return: True si la prédiction a réussi
        """
        if self.series_dict is None:
            # Période plus large que la fenêtre : des jours d'import manquants ne doivent pas
            # empêcher de trouver window_size prix par carte (seuls les derniers sont utilisés)
            self.gatherData(self.window_size * 3)

        # Dernière fenêtre de chaque carte et type (normal et foil)
        # Une fenêtre entièrement à 0 (prix absent) n'est pas prédite : le modèle n'apprend pas sur ces cas
        batch_keys = []
        batch_windows = []
        for product_id, series_type in self.series_dict.items():
            for field, serie_values in series_type.items():
                if len(serie_values) >= self.window_size and serie_values[-self.window_size:].any():
                    batch_keys.append((product_id, field))
                    batch_windows.append(serie_values[-self.window_size:])

        # Préparation des résultats par jour
        all_results = []
        if batch_windows:
            preds = self.forecast(batch_windows, self.nb_predictions)
            for i in range(self.nb_predictions):
                result_dict = {}
                for idx, (product_id, field) in enumerate(batch_keys):
                    pred = preds[idx, i]
                    if product_id not in result_dict:
                        result_dict[product_id] = {
                            'id': '0',
                            'date_data': self.maxDate + pd.Timedelta(days=i+1),
                            'idProduct': product_id,
                            'avg1': None,
                            'avg1_foil': None
                        }
                    # Si la prédiction est négative ou nulle, on la considère comme absente
                    if pred <= 0:
                        result_dict[product_id][field] = None
                    else:
                        result_dict[product_id][field] = float(pred)
                # Ajoute tous les résultats de ce jour
                all_results.extend(result_dict.values())

        # Sans prévision, on s'arrête avant que insertPredictionsToDB ne vide la table
        if not all_results:
            raise RuntimeError(f"Aucune prévision générée : aucune carte n'a {self.window_size} prix récents en base")

        # Écriture groupée du CSV
        with open(self.temp_folder+"predicts.csv", "w", encoding="utf-8") as f_predicts:
            for data in all_results:
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