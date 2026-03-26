"""
SentinelAI v2 — FastAPI Backend (FINAL)
Run: uvicorn main:app --reload --port 8000
"""

import os, time, json
import numpy as np
import joblib
from xgboost import XGBClassifier
import lightgbm as lgb
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

from sentinel_utils import risk_decision, get_llm_explanation


# ── Features ───────────────────────────────────────────────
FEATURES_V2 = [
    "amount_log", "amount_z", "hour_sin", "hour_cos",
    "is_online", "high_amt", "very_high_amt", "night_txn",
    "category_online_risk", "geo_risk",
    "V1","V2","V3","V4","V5","V6","V7","V8","V9","V10",
    "V11","V12","V13","V14","V15","V16","V17","V18","V19","V20",
    "V21","V22","V23","V24","V25","V26","V27","V28",
    "bal_delta", "has_pca",
]


# ── App ────────────────────────────────────────────────────
app = FastAPI(
    title="SentinelAI v2",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Paths ──────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "sentinel_model.json")
SCALER_PATH = os.path.join(BASE_DIR, "sentinel_scaler.pkl")
ENSEMBLE_PATH = os.path.join(BASE_DIR, "sentinel_ensemble.pkl")
META_PATH = os.path.join(BASE_DIR, "model_meta.json")


# ── Load model ─────────────────────────────────────────────
model = XGBClassifier()
model.load_model(MODEL_PATH)

scaler = joblib.load(SCALER_PATH)

ensemble = None
try:
    if os.path.exists(ENSEMBLE_PATH):
        ensemble = joblib.load(ENSEMBLE_PATH)
        print("[SentinelAI] Ensemble loaded.")
    else:
        print("[SentinelAI] XGBoost-only mode.")
except Exception as e:
    print("[SentinelAI] Ensemble failed:", e)
    ensemble = None


DEFAULT_THRESHOLD = 0.5
if os.path.exists(META_PATH):
    with open(META_PATH) as f:
        meta = json.load(f)
        DEFAULT_THRESHOLD = meta.get("threshold", 0.5)


# ── Risk mappings ──────────────────────────────────────────
CATEGORY_RISK = {
    "shopping_net": 1,
    "misc_net": 1,
    "travel": 1,
}

HIGH_RISK_LOCATIONS = ["Unknown", "VPN", "International"]


# ── Schema ────────────────────────────────────────────────
class TransactionRequest(BaseModel):
    amount: float
    hour: Optional[int] = None
    time: Optional[float] = None

    device: Optional[str] = "Unknown"
    location: Optional[str] = "Unknown"
    category: Optional[str] = "misc_pos"

    # PCA
    V1: Optional[float] = 0.0; V2: Optional[float] = 0.0
    V3: Optional[float] = 0.0; V4: Optional[float] = 0.0
    V5: Optional[float] = 0.0; V6: Optional[float] = 0.0
    V7: Optional[float] = 0.0; V8: Optional[float] = 0.0
    V9: Optional[float] = 0.0; V10: Optional[float] = 0.0
    V11: Optional[float] = 0.0; V12: Optional[float] = 0.0
    V13: Optional[float] = 0.0; V14: Optional[float] = 0.0
    V15: Optional[float] = 0.0; V16: Optional[float] = 0.0
    V17: Optional[float] = 0.0; V18: Optional[float] = 0.0
    V19: Optional[float] = 0.0; V20: Optional[float] = 0.0
    V21: Optional[float] = 0.0; V22: Optional[float] = 0.0
    V23: Optional[float] = 0.0; V24: Optional[float] = 0.0
    V25: Optional[float] = 0.0; V26: Optional[float] = 0.0
    V27: Optional[float] = 0.0; V28: Optional[float] = 0.0

    bal_delta: Optional[float] = 0.0
    simulate_fraud: Optional[bool] = False


class PredictResponse(BaseModel):
    fraud_probability: float
    risk_score: float
    decision: str
    label: str
    message: str
    explanation: str
    latency_ms: float


# ── Feature builder ───────────────────────────────────────
def build_features_v2(req: TransactionRequest):

    hour = req.hour if req.hour is not None else datetime.utcnow().hour
    hour = int(hour) % 24

    amount = req.amount

    # simulate fraud
    if req.simulate_fraud:
        amount *= 5
        hour = np.random.choice([1,2,3,23])

    # time encoding
    rad = 2 * np.pi * hour / 24

    # safe scaler
    try:
        amount_z = scaler.transform([[amount]])[0][0]
    except:
        amount_z = 0.0

    is_online = 1 if "online" in (req.device or "").lower() else 0

    category_risk = CATEGORY_RISK.get(req.category, 0)

    geo_risk = 1 if any(x.lower() in (req.location or "").lower()
                        for x in HIGH_RISK_LOCATIONS) else 0

    v_vals = {f"V{i}": getattr(req, f"V{i}", 0.0) or 0.0 for i in range(1,29)}
    has_pca = 1 if any(abs(v) > 1e-6 for v in v_vals.values()) else 0

    row = {
        "amount_log": np.log1p(amount),
        "amount_z": amount_z,
        "hour_sin": np.sin(rad),
        "hour_cos": np.cos(rad),

        "is_online": is_online,
        "high_amt": 1 if amount > 500 else 0,
        "very_high_amt": 1 if amount > 2000 else 0,
        "night_txn": 1 if hour in [23,0,1,2,3,4] else 0,

        "category_online_risk": category_risk,
        "geo_risk": geo_risk,

        "bal_delta": req.bal_delta or 0.0,
        "has_pca": has_pca,

        **v_vals,
    }

    return np.array([[row[f] for f in FEATURES_V2]], dtype=np.float32)


# ── Routes ────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "running", "ensemble": ensemble is not None}


@app.get("/health")
def health():
    return {"status": "ok", "ensemble": ensemble is not None}


@app.post("/predict", response_model=PredictResponse)
async def predict(req: TransactionRequest):

    t0 = time.perf_counter()
    features = build_features_v2(req)

    # ── Model prediction ──
    if ensemble:
        p_xgb = model.predict_proba(features)[0,1]
        p_lgb = ensemble["lgb"].predict_proba(features)[0,1]
        fraud_prob = ensemble["meta"].predict_proba([[p_xgb, p_lgb]])[0,1]
    else:
        fraud_prob = model.predict_proba(features)[0,1]

    # ── Rule-based override ──
    if req.amount > 100000 and fraud_prob > 0.3:
        fraud_prob = min(1.0, fraud_prob + 0.25)

    risk_score = round(fraud_prob * 100, 2)
    verdict = risk_decision(risk_score)

    # ── LLM explanation (safe fallback) ──
    try:
        explanation = await get_llm_explanation(
            amount=req.amount,
            time=req.time or 0,
            device=req.device,
            location=req.location,
            risk_score=risk_score,
            decision=verdict["decision"],
        )
    except:
        explanation = f"Risk score {risk_score}. Decision: {verdict['decision']}."

    latency = round((time.perf_counter() - t0)*1000, 2)

    return PredictResponse(
        fraud_probability=round(float(fraud_prob), 6),
        risk_score=risk_score,
        decision=verdict["decision"],
        label=verdict["label"],
        message=verdict["message"],
        explanation=explanation,
        latency_ms=latency,
    )