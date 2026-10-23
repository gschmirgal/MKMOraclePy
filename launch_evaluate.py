# MKMOraclePi Evaluation Script
# Compare le modèle à la baseline naïve "le prix reste égal au dernier prix connu".
# Les nb_predictions derniers jours de chaque série sont mis de côté, le modèle est
# entraîné sur le reste (sans écraser le modèle sauvegardé), puis on compare les erreurs.
#
# Usage : python launch_evaluate.py [nombre_de_jours_d_historique]
import sys

import numpy as np

from src.predictopi import predictopi

limit = int(sys.argv[1]) if len(sys.argv) > 1 else None

ia = predictopi(reset=True)
series_dict = ia.gatherData(limit)
horizon = ia.nb_predictions
window_size = ia.window_size

# Découpage temporel : apprentissage sur le début des séries, test sur la fin
train_dict = {}
windows = []
actuals = []
for product_id, fields in series_dict.items():
    train_dict[product_id] = {}
    for field, serie_values in fields.items():
        train_dict[product_id][field] = serie_values[:-horizon]
        if len(serie_values) < window_size + horizon:
            continue
        window = serie_values[-horizon-window_size:-horizon]
        actual = serie_values[-horizon:]
        # Un prix à 0 signifie "absent" : on n'évalue que les séries complètes
        if np.all(window > 0) and np.all(actual > 0):
            windows.append(window)
            actuals.append(actual)

if not windows:
    print("Pas assez de données pour évaluer le modèle.")
    sys.exit(1)

ia.series_dict = train_dict
if not ia.learn(save=False):
    print("Pas assez de données pour entraîner le modèle.")
    sys.exit(1)

windows = np.array(windows, dtype=np.float32)
actuals = np.array(actuals, dtype=np.float32)
model_preds = ia.forecast(windows, horizon)
naive_preds = np.repeat(windows[:, -1:], horizon, axis=1)

model_err = np.abs(model_preds - actuals)
naive_err = np.abs(naive_preds - actuals)

print(f"Séries évaluées : {len(windows)} (fenêtre {window_size} j, horizon {horizon} j)\n")
print(f"{'Jour':>4} | {'MAE modèle':>10} | {'MAE naïf':>10} | {'MAPE modèle':>11} | {'MAPE naïf':>10} | {'modèle meilleur':>15}")
for day in range(horizon):
    print(
        f"J+{day+1:<2} | "
        f"{model_err[:, day].mean():>10.3f} | "
        f"{naive_err[:, day].mean():>10.3f} | "
        f"{(model_err[:, day] / actuals[:, day]).mean():>10.1%} | "
        f"{(naive_err[:, day] / actuals[:, day]).mean():>9.1%} | "
        f"{(model_err[:, day] < naive_err[:, day]).mean():>15.1%}"
    )
print(
    f"\nTotal | MAE modèle {model_err.mean():.3f} € vs naïf {naive_err.mean():.3f} € | "
    f"MAPE modèle {(model_err / actuals).mean():.1%} vs naïf {(naive_err / actuals).mean():.1%}"
)
