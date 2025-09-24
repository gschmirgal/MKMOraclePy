# MKMOraclePi Launch Script
from src.predictopi import predictopi

ia = predictopi()

ia.predict()
ia.insertPredictionsToDB()
