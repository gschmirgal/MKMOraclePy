# MKMOraclePi Launch Script
import sys
import traceback

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

except Exception:
    traceback.print_exc()
    log.setFailed()
    sys.exit(1)
