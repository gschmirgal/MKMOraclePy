# 🔮 MKMOraclePi - Prédicteur de Prix Cardmarket

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python&logoColor=white)
![MySQL](https://img.shields.io/badge/MySQL-5.7+-orange?style=for-the-badge&logo=mysql&logoColor=white)
![Machine Learning](https://img.shields.io/badge/ML-RandomForest-green.svg?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**MKMOraclePi** est un système de prédiction intelligent qui analyse et prédit l'évolution des prix des cartes Magic: The Gathering vendues sur le marketplace Cardmarket.

## 🎯 **Objectif**

Ce projet utilise des techniques de machine learning (RandomForest) pour analyser les historiques de prix et prédire les tendances futures des cartes Magic, permettant aux collectionneurs et investisseurs de prendre des décisions éclairées.

## ✨ **Fonctionnalités**

- 📊 **Collecte de données** : Extraction des historiques de prix depuis une base de données
- 🤖 **Intelligence Artificielle** : Modèle RandomForest optimisé pour les séries temporelles
- 📈 **Prédictions** : Prévision des prix sur 7 jours (normal et foil)
- 💾 **Persistance** : Sauvegarde automatique des modèles entraînés
- 🗄️ **Base de données** : Insertion automatique des prédictions en base
- ⚡ **Optimisé mémoire** : Gestion efficace des grandes séries temporelles

## 🏗️ **Architecture**

```
MKMOraclePi/
├── src/
│   ├── predictopi.py    # Classe principale de prédiction
│   ├── common.py        # Fonctions utilitaires
│   └── db.py           # Gestion de la base de données
├── data/               # Dossier temporaire (modèles, CSV)
├── launch_learn.py     # Script d'entraînement
├── launch_predict.py   # Script de prédiction
├── config.ini          # Configuration (DB, chemins)
└── requirements.txt    # Dépendances Python
```

## 🚀 **Installation**

### Prérequis
- Python 3.11+
- MySQL 8.0+
- Historique de prix des cartes Magic dans une base de données

### 1. Cloner le projet
```bash
git clone https://github.com/gschmirgal/MKMOraclePi.git
cd MKMOraclePi
```

### 2. Créer l'environnement virtuel
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# ou
source .venv/bin/activate  # Linux/Mac
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 4. Configuration
```bash
cp config.ini.sample config.ini
# Éditer config.ini avec vos paramètres de base de données
```

## 🛠️ Mise à jour de la base de données

Si votre base n'est pas gérée automatiquement par l'ORM du front, vous pouvez utiliser le fichier `update from mkmpy.sql` pour mettre à jour la structure ou les données de la base MySQL.

- Ce fichier contient les requêtes SQL nécessaires pour adapter la base aux besoins du projet MKMOraclePi.
- À exécuter manuellement via un client MySQL ou un outil d'administration si besoin.
- Exemple d'utilisation :

```bash
mysql -u <user> -p <database> < "update from mkmpy.sql"
```

**Structure de la base de données attendue :**

```sql
-- Table des prix historiques
CREATE TABLE prices (
    id INT AUTO_INCREMENT PRIMARY KEY,
    idProduct INT NOT NULL,
    date_data DATE NOT NULL,
    avg1 DECIMAL(10,2),          -- Prix normal
    avg1_foil DECIMAL(10,2)      -- Prix foil
);

-- Table des prédictions
CREATE TABLE prices_predict (
    id INT AUTO_INCREMENT PRIMARY KEY,
    idProduct INT NOT NULL,
    date_data DATE NOT NULL,
    avg1 DECIMAL(10,2),
    avg1_foil DECIMAL(10,2)
);
```

## 📖 **Utilisation**

### Entraînement du modèle
```bash
python launch_learn.py
```

### Génération des prédictions
```bash
python launch_predict.py
```

### Utilisation programmatique
```python
from src.predictopi import predictopi

# Créer une instance
predictor = predictopi(reset=False)

# Collecter les données (7 derniers jours)
predictor.gatherData(limit=7)

# Entraîner le modèle (si nécessaire)
predictor.learn()

# Générer les prédictions
predictor.predict()

# Insérer en base de données
predictor.insertPredictionsToDB()
```

## ⚙️ **Configuration**

Le fichier `config.ini` contient les paramètres suivants :

```ini
[Database]
# Adresse du serveur MySQL (ex: localhost ou IP)
host     = yourhost
# Nom d'utilisateur MySQL
user     = user
# Mot de passe MySQL
password = pwd
# Nom de la base de données à utiliser
database = mkmpy
# Port MySQL (par défaut 3306)
port     = 3306

[Folders]
# Dossier temporaire pour stocker les modèles et fichiers intermédiaires
temp       = ./data/

[IA]
# Nombre d'arbres dans la forêt
n_estimators=200
# Profondeur maximale des arbres
max_depth=15
# Nombre minimum d'échantillons requis pour diviser un noeud
min_samples_split=5
# Nombre minimum d'échantillons requis pour être à une feuille
min_samples_leaf=2
# Nombre de caractéristiques à considérer lors de la recherche de la meilleure séparation
max_features='sqrt'
# Nombre de cœurs CPU à utiliser (-1 pour tous les cœurs)
n_jobs=-1
# État aléatoire pour la reproductibilité des résultats
random_state=42
```

## 🧠 **Modèle de Machine Learning**

- **Algorithme** : RandomForestRegressor (100 arbres)
- **Features** : Fenêtre glissante de 4 jours de prix
- **Normalisation** : StandardScaler pour optimiser les performances
- **Prédiction** : 7 jours futurs par itération
- **Types** : Prix normal et foil traités séparément

### Optimisations mémoire
- Pré-allocation des arrays NumPy
- Utilisation de `float32` au lieu de `float64`
- Nettoyage explicite des variables temporaires
- Fenêtre glissante efficace avec `np.roll()`

## 🚀 Optimisations récentes

### Prédiction en batch
- La méthode `predict` utilise désormais la prédiction en batch pour accélérer le calcul sur de grandes quantités de cartes.
- Les features de toutes les cartes à prédire pour chaque jour sont regroupées et traitées en une seule opération, exploitant le parallélisme interne de scikit-learn.
- Le CSV de sortie est écrit en une seule fois pour limiter les accès disque.

### Gains de performance
- Temps de prédiction fortement réduit, surtout pour les gros volumes de données.
- Moins d'overhead Python, meilleure utilisation du CPU.
- Optimisation mémoire conservée (pré-allocation, float32, nettoyage explicite).

## 🆕 Exemple d'utilisation optimisée

```python
from src.predictopi import predictopi

predictor = predictopi(reset=False)
predictor.gatherData(limit=7)
predictor.learn()
predictor.predict()  # Prédiction accélérée en batch
predictor.insertPredictionsToDB()
```

## 📊 **Données de sortie**

Les prédictions sont sauvegardées dans :
- **Fichier CSV** : `data/predicts.csv`
- **Base de données** : Table `prices_predict`

Format des données :
```csv
id;date_data;avg1;avg1_foil;idProduct
0;2025-09-26;12.34;45.67;12345
```

## 🔧 **Développement**

### Structure du code principal
```python
class predictopi:
    def __init__(self, reset=False)      # Initialisation
    def gatherData(self, limit=None)     # Collecte des données
    def learn(self)                      # Entraînement
    def predict(self)                    # Prédiction
    def insertPredictionsToDB(self)      # Insertion en base
```

## 📈 **Performance**

- **Optimisation mémoire** : -50% d'usage grâce aux optimisations numpy
- **Vitesse** : Pré-allocation des arrays pour éviter les copies
- **Persistance** : Sauvegarde automatique des modèles entraînés

## 📄 **Licence**

Ce projet est sous licence MIT.

## 🛠️ **Stack Technique**

- **Python 3.11+** - Langage principal
- **NumPy** - Calculs numériques optimisés
- **pandas** - Manipulation des données temporelles
- **scikit-learn** - Machine Learning (RandomForest, StandardScaler)
- **joblib** - Sérialisation des modèles
- **MySQL** - Base de données
