from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()

# --------------------------
# Load models (if available)
# --------------------------
try:
    with open("models/model.pkl", "rb") as f:
        clf_model = pickle.load(f)
except:
    clf_model = None

try:
    with open("models/model_goals.pkl", "rb") as f:
        goal_models = pickle.load(f)
except:
    goal_models = None


# --------------------------
# Request Models
# --------------------------
class PredictRequest(BaseModel):
    home: str
    away: str
    datetime: str
    homeLast5For: float
    homeLast5Against: float
    awayLast5For: float
    awayLast5Against: float
    forbetHomePct: float
    forbetDrawPct: float
    forbetAwayPct: float


class GoalRequest(BaseModel):
    homeLast5For: float
    homeLast5Against: float
    awayLast5For: float
    awayLast5Against: float


# --------------------------
# Poisson fallback
# --------------------------
def poisson_predict(lambda_home, lambda_away):
    home = lambda_home / (lambda_home + lambda_away)
    away = lambda_away / (lambda_home + lambda_away)
    draw = 1 - (home + away)
    return home, draw, away


# --------------------------
# Main Prediction API
# --------------------------
@app.post("/api/predict")
def predict(req: PredictRequest):

    if clf_model is None:
        lh = req.homeLast5For / 5
        la = req.homeLast5Against / 5
        rh = req.awayLast5For / 5
        ra = req.awayLast5Against / 5

        h, d, a = poisson_predict(lh, ra)
        return {
            "home": round(h * 100, 2),
            "draw": round(d * 100, 2),
            "away": round(a * 100, 2),
            "recommendation": "HOME" if h > a else "AWAY",
            "model": "poisson-fallback"
        }

    X = np.array([
        req.homeLast5For, req.homeLast5Against,
        req.awayLast5For, req.awayLast5Against
    ]).reshape(1, -1)

    pred = clf_model["model"].predict(X)[0]
    probs = clf_model["model"].predict_proba(X)[0]

    labels = clf_model["model"].classes_
    mapping = {
        label: float(probs[i] * 100)
        for i, label in enumerate(labels)
    }

    return {
        "home": mapping.get("HOME", 0),
        "draw": mapping.get("DRAW", 0),
        "away": mapping.get("AWAY", 0),
        "recommendation": pred,
        "model": "ML-classifier"
    }


# --------------------------
# Goals Prediction API
# --------------------------
@app.post("/api/predict_goals")
def predict_goals(req: GoalRequest):

    if goal_models is None:
        h = req.homeLast5For / 5
        a = req.awayLast5For / 5
        return {
            "predicted_home_goals": round(h, 2),
            "predicted_away_goals": round(a, 2),
            "model": "fallback"
        }

    X = np.array([
        req.homeLast5For, req.homeLast5Against,
        req.awayLast5For, req.awayLast5Against
    ]).reshape(1, -1)

    home_g = goal_models["home_model"].predict(X)[0]
    away_g = goal_models["away_model"].predict(X)[0]

    return {
        "predicted_home_goals": round(float(home_g), 2),
        "predicted_away_goals": round(float(away_g), 2),
        "model": "ml-regression"
    }
