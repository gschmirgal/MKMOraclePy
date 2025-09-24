# MKMOraclePi Launch Script
from src.predictopi import predictopi

ia = predictopi()

# Collecte des données avec limite pour réduire l'usage mémoire
ia.gatherData()
ia.learn()