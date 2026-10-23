# MKMOraclePi

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)
![MySQL](https://img.shields.io/badge/MySQL-8.0-4479A1?logo=mysql&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-F7931E?logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)

Prédiction des prix des cartes Magic: The Gathering vendues sur Cardmarket.

MKMOraclePi lit l'historique des prix en base, entraîne un modèle RandomForest et écrit les prix prévus pour les prochains jours dans la table `prices_predict`.

## Place dans l'écosystème

MKMOraclePi est l'un des trois projets de la stack, assemblés par [dockermkmpy](https://github.com/gschmirgal/dockermkmpy) :

| Projet | Rôle |
|---|---|
| [mkmpy](https://github.com/gschmirgal/mkmpy) | Importe chaque jour les prix Cardmarket dans la table `prices` |
| **MKMOraclePi** | Apprend sur `prices` et écrit les prévisions dans `prices_predict` |
| [frontMkmPy](https://github.com/gschmirgal/frontMkmPy) | Affiche les prix et les prévisions ; gère le schéma de la base (migrations Doctrine) |

Pour un déploiement complet, passer par **dockermkmpy**. La suite décrit une installation autonome.

## Installation

Prérequis :

- Python 3.12 ou plus récent
- Une base MySQL 8.0 déjà alimentée par mkmpy, avec `local_infile` activé côté serveur (les prévisions sont importées par `LOAD DATA LOCAL INFILE`)

```bash
git clone https://github.com/gschmirgal/MKMOraclePy.git
cd MKMOraclePy
python3 -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
cp config.ini.sample config.ini  # puis renseigner la section [Database]
```

Si le schéma n'est pas géré par frontMkmPy, créer les tables propres à MKMOraclePi avec :

```bash
mysql -u <user> -p <database> < "update from mkmpy.sql"
```

## Utilisation

Les scripts se lancent depuis la racine du projet :

| Commande | Rôle | Fréquence conseillée |
|---|---|---|
| `python launch_learn.py` | Entraîne le modèle sur tout l'historique et le sauvegarde | Hebdomadaire |
| `python launch_predict.py` | Prédit les prochains jours et remplace le contenu de `prices_predict` | Quotidienne, après l'import mkmpy |
| `python launch_evaluate.py [jours]` | Mesure la qualité du modèle, sans rien écrire en base ni sur disque | Ponctuelle |

`launch_learn.py` et `launch_predict.py` enregistrent chaque exécution dans la table `logs_oracle`, avec le statut `ongoing`, puis `finished` ou `failed`. En cas d'erreur, ils affichent la trace et sortent avec le code 1, ce qui permet à cron de détecter l'échec.

### Évaluer le modèle

`launch_evaluate.py` met de côté les derniers jours de chaque série (autant que `nb_predictions`), entraîne un modèle temporaire sur le reste, puis compare ses prévisions à la **baseline naïve** « le prix reste égal au dernier prix connu ». L'argument facultatif limite l'historique utilisé aux N derniers jours.

```
Jour | MAE modèle |   MAE naïf | MAPE modèle |  MAPE naïf | modèle meilleur
J+1  |      1.267 |      1.130 |       2.6% |      2.3% |           40.8%
...
```

Si le modèle ne fait pas mieux que la baseline, ses prévisions n'apportent rien.

## Configuration

Le fichier `config.ini` est cherché dans le dossier parent du dossier courant (`../config.ini`), puis dans le dossier courant. Voir [config.ini.sample](config.ini.sample) pour un exemple commenté.

| Section | Clé | Défaut | Rôle |
|---|---|---|---|
| `Database` | `host`, `port`, `user`, `password`, `database` | — | Connexion MySQL |
| `Folders` | `temp` | `./data/` | Dossier du modèle sauvegardé et du CSV de prévisions |
| `IA` | `window_size` | `7` | Nombre de jours de prix en entrée du modèle |
| `IA` | `nb_predictions` | `10` | Nombre de jours prédits |
| `IA` | `n_estimators` | `200` | Nombre d'arbres |
| `IA` | `max_depth` | `15` | Profondeur maximale des arbres |
| `IA` | `min_samples_split` | `5` | Échantillons minimum pour diviser un nœud |
| `IA` | `min_samples_leaf` | `2` | Échantillons minimum par feuille |
| `IA` | `max_features` | `sqrt` | Features considérées à chaque division (sans guillemets) |
| `IA` | `n_jobs` | `-1` | Cœurs CPU utilisés (`-1` : tous) |
| `IA` | `random_state` | `42` | Graine aléatoire, pour des résultats reproductibles |

Après un changement de `window_size`, il faut relancer `launch_learn.py` : le modèle sauvegardé attend une fenêtre de la taille avec laquelle il a été entraîné.

## Fonctionnement

1. **Données** : pour chaque carte, la série des prix moyens `avg1` (normal) et `avg1_foil` (foil) est lue dans `prices`, triée par date. Un prix absent (`NULL`) est remplacé par 0.
2. **Apprentissage** : chaque série est découpée en fenêtres glissantes de `window_size` jours. Le modèle apprend à prédire le prix du jour suivant à partir d'une fenêtre. Un seul modèle est entraîné pour toutes les cartes, normales et foil confondues.
3. **Prédiction** : le modèle prédit J+1 à partir des derniers jours connus, réinjecte cette prévision dans la fenêtre pour prédire J+2, et ainsi de suite jusqu'à `nb_predictions`. Une prévision négative ou nulle est enregistrée comme `NULL`.
4. **Écriture** : les prévisions sont écrites dans `<temp>/predicts.csv`, puis la table `prices_predict` est vidée et rechargée depuis ce fichier.

Fichiers produits dans le dossier `temp` :

| Fichier | Contenu |
|---|---|
| `rf_model_mem.joblib` | Modèle entraîné |
| `scaler_X.joblib` | Normaliseur des features |
| `predicts.csv` | Dernières prévisions (`id;date_data;avg1;avg1_foil;idProduct`) |

### Limites connues

- Le modèle travaille sur des prix absolus, de quelques centimes à plusieurs centaines d'euros. Une RandomForest ne sait pas extrapoler au-delà des prix vus à l'entraînement.
- Les prix absents remplacés par 0 et les jours manquants dans l'historique faussent les fenêtres.
- Les erreurs s'accumulent d'un jour prédit au suivant.

`launch_evaluate.py` permet de vérifier l'effet de chaque amélioration sur ces points.

## Structure

```
MKMOraclePy/
├── launch_learn.py         # Entraînement
├── launch_predict.py       # Prédiction et écriture en base
├── launch_evaluate.py      # Évaluation face à la baseline naïve
├── src/
│   ├── predictopi.py       # Classe predictopi : données, apprentissage, prédiction
│   ├── db.py               # Connexion MySQL
│   ├── log.py              # Journalisation dans logs_oracle
│   └── common.py           # Lecture de config.ini, export CSV
├── data/                   # Dossier temporaire par défaut
├── config.ini.sample
├── update from mkmpy.sql   # Tables propres à MKMOraclePi
└── requirements.txt
```

## Licence

MIT
