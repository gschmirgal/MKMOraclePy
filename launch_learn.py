# MKMOraclePi Launch Script
from src.predictopi import predictopi
from src.log import log

# Initialise le système de log et crée une nouvelle entrée de log
log = log()
logId = log.createLogEntry("learn")

try:
    log.setStatus("ongoing")
    ia = predictopi(reset=False)
    # Collecte des données avec limite pour réduire l'usage mémoire
    ia.gatherData()
    ia.learn()
    log.setStatus("finished")
    
except Exception as e:
    print(f"Error during processing: {e}")
    log.setStatus("finished")