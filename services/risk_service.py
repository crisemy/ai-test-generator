import os
from typing import Any, Dict, List, Tuple, Union

import joblib
import pandas as pd


FeatureInput = Union[str, Dict[str, Any]]


def load_risk_model(model_path: str = "data/risk_model.pkl", encoder_path: str = "data/label_encoder.pkl"):
    if not (os.path.exists(model_path) and os.path.exists(encoder_path)):
        raise FileNotFoundError("ML model files not found. Please run the training notebook first.")

    clf = joblib.load(model_path)
    le = joblib.load(encoder_path)
    return clf, le


def extract_features(tc: FeatureInput) -> Dict[str, int]:
    if isinstance(tc, str):
        text = tc.lower()
        neg = 0
        edge = 0
    else:
        text = " ".join([str(tc.get(k, "")) for k in ["scenario", "given", "when", "then"]]).lower()
        neg = 1 if tc.get("type") == "Negative" else 0
        edge = 1 if tc.get("type") == "Edge" else 0

    return {
        "loc": len(text) // 10,
        "cyclomatic_complexity": len(text.split()) // 5 + 5,
        "prev_defects": 4 if any(w in text for w in ["transfer", "payment", "money", "withdraw"]) else 1,
        "negative_tests": neg,
        "edge_tests": edge,
        "money_related": 1 if any(w in text for w in ["amount", "money", "transfer", "currency", "$"]) else 0,
        "security_related": 1 if any(w in text for w in ["2fa", "password", "auth", "security", "token", "otp"]) else 0,
    }


def score_backlog(df_backlog: pd.DataFrame, story_col: str, model_rf: Any, label_encoder: Any) -> pd.DataFrame:
    scores: List[float] = []
    labels: List[str] = []

    for text in df_backlog[story_col].fillna(""):
        feats = extract_features(str(text))
        x_pred = pd.DataFrame([feats])
        pred_encoded = model_rf.predict(x_pred)[0]
        risk_prob = max(model_rf.predict_proba(x_pred)[0]) * 100
        scores.append(round(risk_prob, 1))
        labels.append(label_encoder.inverse_transform([pred_encoded])[0])

    scored = df_backlog.copy()
    scored["Risk Score"] = scores
    scored["Risk Label"] = labels
    return scored.sort_values(by="Risk Score", ascending=False)


def apply_risk_scoring(test_cases: List[Dict[str, Any]], model_rf: Any, label_encoder: Any) -> List[Dict[str, Any]]:
    scored_cases: List[Dict[str, Any]] = []

    for tc in test_cases:
        feats = extract_features(tc)
        x_pred = pd.DataFrame([feats])
        pred_encoded = model_rf.predict(x_pred)[0]
        risk_label = label_encoder.inverse_transform([pred_encoded])[0]
        risk_proba = model_rf.predict_proba(x_pred)[0]
        risk_score = max(risk_proba) * 100

        tc_copy = dict(tc)
        tc_copy["risk_score"] = round(risk_score, 1)
        tc_copy["risk_label"] = risk_label
        scored_cases.append(tc_copy)

    return sorted(scored_cases, key=lambda x: x.get("risk_score", 0), reverse=True)
