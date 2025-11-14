# backend/app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import math
import numpy as np

app = FastAPI(title="SmartPredictor — Hybrid Inline Model")


# ---------------------------------------------------
# Data models
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
# Math utils
# ---------------------------------------------------
def safe_div(a, b, default=0.0):
    try:
        return a / b if b != 0 else default
    except:
        return default


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
# Step 1 — Strength calculation
# ---------------------------------------------------
def compute_strengths(h_for, h_against, a_for, a_against, baseline=1.25):
    h_avg_for = safe_div(h_for, 5)
    h_avg_against = safe_div(h_against, 5)
    a_avg_for = safe_div(a_for, 5)
    a_avg_against = safe_div(a_against, 5)

    h_attack = safe_div(h_avg_for, baseline)
    h_defense = safe_div(h_avg_against, baseline)
    a_attack = safe_div(a_avg_for, baseline)
    a_defense = safe_div(a_avg_against, baseline)

    return {
        "h_attack": h_attack,
        "h_defense": h_defense,
        "a_attack": a_attack,
        "a_defense": a_defense
    }


# ---------------------------------------------------
# Step 2 — Expected goals (lambda)
# ---------------------------------------------------
def expected_goals(strengths, home_adv=1.10):
    λh = (strengths["h_attack"] * (1.0 / max(strengths["a_defense"], 0.01))) * home_adv
    λa = (strengths["a_attack"] * (1.0 / max(strengths["h_defense"], 0.01)))
    return round(max(0.05, λh), 3), round(max(0.05, λa), 3)


# ---------------------------------------------------
# Step 3 — Score probabilities matrix
# ---------------------------------------------------
def scoreline_probs(lambda_h, lambda_a, max_goals=6):
    home_dist = [poisson(i, lambda_h) for i in range(0, max_goals + 1)]
    away_dist = [poisson(i, lambda_a) for i in range(0, max_goals + 1)]
    home_dist = normalize(home_dist)
    away_dist = normalize(away_dist)

    scores = {}
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            scores[f"{h}-{a}"] = home_dist[h] * away_dist[a]

    return scores


# ---------------------------------------------------
# Step 4 — Home / Draw / Away + BTTS
# ---------------------------------------------------
def aggregate(scores):
    p_home = p_draw = p_away = 0.0
    p_btts = 0.0

    for sc, p in scores.items():
        h, a = map(int, sc.split('-'))

        if h > a:
            p_home += p
        elif h == a:
            p_draw += p
        else:
            p_away += p

        if h > 0 and a > 0:
            p_btts += p

    return {
        "home": p_home,
        "draw": p_draw,
        "away": p_away,
        "btts": p_btts
    }


# ---------------------------------------------------
# Step 5 — Combine with odds
# ---------------------------------------------------
def combine_probs(model, ext):
    mh, md, ma = model
    eh, ed, ea = ext

    if (eh + ed + ea) == 0:
        return model

    s = mh + md + ma
    if s == 0:
        return model

    # normalize ext
    se = eh + ed + ea
    eh, ed, ea = eh / se, ed / se, ea / se

    α = 0.7  # model weight
    home = α * mh + (1 - α) * eh
    draw = α * md + (1 - α) * ed
    away = α * ma + (1 - α) * ea

    n = home + draw + away
    return home / n, draw / n, away / n


# ---------------------------------------------------
# Step 6 — Confidence, risk, fair odds
# ---------------------------------------------------
def evaluate_risk(probs):
    home, draw, away = probs
    top_two = sorted([home, draw, away], reverse=True)

    gap = top_two[0] - top_two[1]
    confidence = 35 + gap * 65

    if confidence < 45:
        risk = "HIGH"
    elif confidence < 60:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return round(confidence, 2), risk


# ---------------------------------------------------
# API — Predict Winner & Full Analysis
# ---------------------------------------------------
@app.post("/api/predict")
def predict(req: PredictRequest):

    # strengths
    s = compute_strengths(
        req.homeLast5For,
        req.homeLast5Against,
        req.awayLast5For,
        req.awayLast5Against,
    )

    # lambda goals
    λh, λa = expected_goals(s)

    # score matrix
    matrix = scoreline_probs(λh, λa)
    agg = aggregate(matrix)

    # internal model probs
    m_home = agg["home"]
    m_draw = agg["draw"]
    m_away = agg["away"]

    # external odds (convert to decimals)
    eh = (req.forbetHomePct or 0) / 100
    ed = (req.forbetDrawPct or 0) / 100
    ea = (req.forbetAwayPct or 0) / 100

    combined = combine_probs(
        (m_home, m_draw, m_away),
        (eh, ed, ea)
    )

    home_p, draw_p, away_p = combined

    # confidence + risk
    confidence, risk = evaluate_risk((home_p, draw_p, away_p))

    # recommendation
    idx = np.argmax([home_p, draw_p, away_p])
    rec = ["HOME", "DRAW", "AWAY"][idx]

    # fair odds
    fair = round(1 / [home_p, draw_p, away_p][idx], 2)

    # top scorelines
    top = sorted(matrix.items(), key=lambda x: x[1], reverse=True)[:8]
    top_scores = [{"score": s, "prob": round(float(p), 4)} for s, p in top]

    return {
        "home": round(home_p * 100, 2),
        "draw": round(draw_p * 100, 2),
        "away": round(away_p * 100, 2),
        "recommendation": rec,
        "confidence": confidence,
        "risk": risk,
        "fairOdds": fair,
        "btts": round(agg["btts"] * 100, 2),
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

    s = compute_strengths(
        req.homeLast5For, req.homeLast5Against,
        req.awayLast5For, req.awayLast5Against
    )
    λh, λa = expected_goals(s)

    return {
        "predicted_home_goals": λh,
        "predicted_away_goals": λa,
        "model": "hybrid-inline"
    }


# ---------------------------------------------------
# Test endpoint
# ---------------------------------------------------
@app.get("/")
def index():
    return {"status": "SmartPredictor Hybrid Model is running"}
