# MKMOraclePi Launch Script
from src.predictopi import predictopi
from src.log import log

# Initialise le système de log et crée une nouvelle entrée de log
log = log()
log.createLogEntry("predict")

try:
    ia = predictopi()
    ia.predict()
    ia.insertPredictionsToDB()
    log.setStatus("finished")

except Exception as e:
    print(f"Error during processing: {e}")
    log.setStatus("finished")