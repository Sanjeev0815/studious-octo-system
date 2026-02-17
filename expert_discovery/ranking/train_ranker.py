import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "ranking"
DATA_DIR.mkdir(parents=True, exist_ok=True)
sys.path.append(str(BASE_DIR))

from expert_discovery.ranking.feature_engineering import build_training_data
from expert_discovery.utils.metrics import (
    precision_at_k,
    recall_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
)


def evaluate_groups(df, scores, k=3):
    metrics = {"precision": [], "recall": [], "mrr": [], "ndcg": []}
    df = df.copy()
    df["score"] = scores

    for _, group in df.groupby("question_id"):
        group_sorted = group.sort_values("score", ascending=False)
        relevances = group_sorted["label"].tolist()
        metrics["precision"].append(precision_at_k(relevances, k))
        metrics["recall"].append(recall_at_k(relevances, k))
        metrics["mrr"].append(mean_reciprocal_rank(relevances))
        metrics["ndcg"].append(ndcg_at_k(relevances, k))

    return {key: float(np.mean(values)) if values else 0.0 for key, values in metrics.items()}


def main():
    rows = build_training_data(negatives_per_positive=3)
    if not rows:
        raise RuntimeError("No training data generated.")

    df = pd.DataFrame(rows)
    features = ["semantic", "tag_similarity", "graph_score", "expertise", "activity"]

    X = df[features]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    scores = model.predict_proba(X_test)[:, 1]
    df_test = df.loc[X_test.index].copy()
    df_test["label"] = y_test.values

    metrics = evaluate_groups(df_test, scores, k=3)

    curves = {"k": [], "precision": [], "recall": [], "mrr": [], "ndcg": []}
    for k in range(1, 11):
        values = evaluate_groups(df_test, scores, k=k)
        curves["k"].append(k)
        for key in ["precision", "recall", "mrr", "ndcg"]:
            curves[key].append(values[key])

    curve_path = DATA_DIR / "ranking_metrics.png"
    plt.figure(figsize=(8, 5))
    plt.plot(curves["k"], curves["precision"], label="Precision@K")
    plt.plot(curves["k"], curves["recall"], label="Recall@K")
    plt.plot(curves["k"], curves["mrr"], label="MRR")
    plt.plot(curves["k"], curves["ndcg"], label="NDCG")
    plt.xlabel("K")
    plt.ylabel("Score")
    plt.title("Ranking Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(curve_path)

    model_path = MODEL_DIR / "ranker_logistic.joblib"
    joblib.dump(model, model_path)

    metrics_path = DATA_DIR / "metrics_summary.json"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print("Model trained.")
    print(metrics)


if __name__ == "__main__":
    main()
