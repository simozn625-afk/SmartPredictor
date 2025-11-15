from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import math
import numpy as np
import csv
import os

app = FastAPI(title="SmartPredictor — Hybrid Inline Model")

# ---------------------------------------------------
# CSV Reader
# ---------------------------------------------------
def read_csv_simple(path: str):
    if not os.path.exists(path):
        return {"error": f"File not found: {path}"}
    data = []
    with open(path, newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

# ---------------------------------------------------
# Request Models
# ---------------------------------------------------
class PredictRequest(BaseModel):
    home: str
    away: str
    datetime: Optional[str] = None
    homeLast5For: float
    homeLast5Against: float
    awayLast5For: float
    awayLast5Against: float
    forbetHomePct: Optional[float] = 0.0
    forbetDrawPct: Optional[float] = 0.0
    forbetAwayPct: Optional[float] = 0.0

class PredictGoalsRequest(BaseModel):
    homeLast5For: float
    homeLast5Against: float
    awayLast5For: float
    awayLast5Against: float

# ---------------------------------------------------
# Math Helpers
# ---------------------------------------------------
def safe_div(a, b, default=0.0):
    return a / b if b != 0 else default

def poisson(k: int, lamb: float):
    if lamb <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lamb) * (lamb ** k) / math.factorial(k)

def normalize(x: List[float]):
    s = sum(x)
    if s == 0:
        return [1 / len(x)] * len(x)
    return [v / s for v in x]

# ---------------------------------------------------
# Step 1 — Compute Strength
# ---------------------------------------------------
def compute_strengths(h_for, h_against, a_for, a_against, baseline=1.25):
    return {
        "h_attack": safe_div(h_for / 5, baseline),
        "h_defense": safe_div(h_against / 5, baseline),
        "a_attack": safe_div(a_for / 5, baseline),
        "a_defense": safe_div(a_against / 5, baseline)
    }

# ---------------------------------------------------
# Step 2 — λ (Expected Goals)
# ---------------------------------------------------
def expected_goals(s, home_adv=1.10):
    λh = max(0.05, s["h_attack"] / max(s["a_defense"], 0.01) * home_adv)
    λa = max(0.05, s["a_attack"] / max(s["h_defense"], 0.01))
    return round(λh, 3), round(λa, 3)

# ---------------------------------------------------
# Step 3 — Score Probability Matrix
# ---------------------------------------------------
def scoreline_probs(λh, λa, max_goals=6):
    home_dist = normalize([poisson(i, λh) for i in range(max_goals + 1)])
    away_dist = normalize([poisson(i, λa) for i in range(max_goals + 1)])
    scores = {}
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            scores[f"{h}-{a}"] = home_dist[h] * away_dist[a]
    return scores

# ---------------------------------------------------
# Step 4 — Outcome Probabilities
# ---------------------------------------------------
def aggregate(scores):
    p_home = p_draw = p_away = p_btts = 0.0
    for sc, p in scores.items():
        h, a = map(int, sc.split('-'))
        if h > a: p_home += p
        elif h == a: p_draw += p
        else: p_away += p
        if h > 0 and a > 0: p_btts += p
    return {"home": p_home, "draw": p_draw, "away": p_away, "btts": p_btts}

# ---------------------------------------------------
# Step 5 — Combine with bookmaker odds
# ---------------------------------------------------
def combine_probs(model, ext):
    mh, md, ma = model
    eh, ed, ea = ext
    if (eh + ed + ea) == 0:
        return model
    α = 0.7
    home = α*mh + (1-α)*(eh/(eh+ed+ea))
    draw = α*md + (1-α)*(ed/(eh+ed+ea))
    away = α*ma + (1-α)*(ea/(eh+ed+ea))
    n = home + draw + away
    return home/n, draw/n, away/n

# ---------------------------------------------------
# Step 6 — Confidence & Risk
# ---------------------------------------------------
def evaluate_risk(probs):
    home, draw, away = probs
    t = sorted([home, draw, away], reverse=True)
    gap = t[0] - t[1]
    conf = 35 + gap * 65
    risk = "LOW" if conf >= 60 else "MEDIUM" if conf >= 45 else "HIGH"
    return round(conf, 2), risk

# ---------------------------------------------------
# API — Full Prediction
# ---------------------------------------------------
@app.post("/api/predict")
def predict(req: PredictRequest):
    s = compute_strengths(req.homeLast5For, req.homeLast5Against,
                          req.awayLast5For, req.awayLast5Against)

    λh, λa = expected_goals(s)
    matrix = scoreline_probs(λh, λa)
    agg = aggregate(matrix)

    model = (agg["home"], agg["draw"], agg["away"])
    ext = ((req.forbetHomePct or 0)/100,
           (req.forbetDrawPct or 0)/100,
           (req.forbetAwayPct or 0)/100)

    home_p, draw_p, away_p = combine_probs(model, ext)
    confidence, risk = evaluate_risk((home_p, draw_p, away_p))
    rec_idx = int(np.argmax([home_p, draw_p, away_p]))
    rec = ["HOME", "DRAW", "AWAY"][rec_idx]

    top = sorted(matrix.items(), key=lambda x: x[1], reverse=True)[:8]
    top_scores = [{"score": s, "prob": round(float(p), 4)} for s, p in top]

    return {
        "home": round(home_p*100, 2),
        "draw": round(draw_p*100, 2),
        "away": round(away_p*100, 2),
        "recommendation": rec,
        "confidence": confidence,
        "risk": risk,
        "fairOdds": round(1 / [home_p, draw_p, away_p][rec_idx], 2),
        "btts": round(agg["btts"]*100, 2),
        "lambda_home": λh,
        "lambda_away": λa,
        "top_scorelines": top_scores,
        "suggested_final_score": f"{int(round(λh))}-{int(round(λa))}"
    }

# ---------------------------------------------------
# API — Predict Goals Only
# ---------------------------------------------------
@app.post("/api/predict_goals")
def predict_goals(req: PredictGoalsRequest):
    s = compute_strengths(req.homeLast5For, req.homeLast5Against,
                          req.awayLast5For, req.awayLast5Against)
    λh, λa = expected_goals(s)
    return {"predicted_home_goals": λh, "predicted_away_goals": λa}

# ---------------------------------------------------
# API — Import CSV
# ---------------------------------------------------
@app.get("/api/import_csv")
def import_csv(path: str):
    data = read_csv_simple(path)
    return {"count": len(data) if isinstance(data, list) else 0, "rows": data}

# ---------------------------------------------------
# Root
# ---------------------------------------------------
@app.get("/")
def index():
    return {"status": "SmartPredictor is running"}
