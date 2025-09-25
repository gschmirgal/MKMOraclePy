# MKMOraclePi Launch Script
from src.predictopi import predictopi

ia = predictopi(reset=True)

# Collecte des données avec limite pour réduire l'usage mémoire
ia.gatherData()
ia.learn()