"""
Multi-model trainer (Python 3.13+).
- Global StandardScaler once (saved).
- 70/10/20 split (train/val/test).
- Early stopping for LGBM/XGBoost via val set.
- Per-model artifacts: metrics.json, classification_report.txt, confusion_matrix.png, roc_curve.png,
  feature_importance.csv/.png (when supported).
- Robust to sklearn API changes for CalibratedClassifierCV.
- Robust to non-numeric columns (e.g., 'url') in features_extracted.csv.
"""

from pathlib import Path
from datetime import datetime, timezone
import json
import warnings

import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report,
    ConfusionMatrixDisplay, RocCurveDisplay
)
from sklearn.utils.class_weight import compute_class_weight

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

# Optional
try:
    import lightgbm as lgb
except Exception:
    lgb = None
try:
    import xgboost as xgb
except Exception:
    xgb = None

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------- Config ----------------
CONFIG = {
    "data_csv": "features_extracted.csv",
    "label_col": "label",
    "models_dir": "models",
    "results_dir": "results",
    "random_state": 42,
    "train_ratio": 0.70,  # train
    "val_ratio":   0.10,  # validation
    "test_ratio":  0.20,  # test
}

# ---------------- Models ----------------
def build_models(rs: int):
    models = {
        "Logistic Regression": LogisticRegression(
            solver="liblinear", random_state=rs, max_iter=2000, class_weight="balanced"
        ),
        "Calibrated LinearSVC": CalibratedClassifierCV(
            estimator=LinearSVC(dual="auto", random_state=rs, max_iter=5000),
            cv=3
        ),
        "Gaussian Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=15, random_state=rs, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=rs, n_jobs=-1, class_weight="balanced", max_depth=None
        ),
        "AdaBoost": AdaBoostClassifier(
            random_state=rs, n_estimators=400, learning_rate=0.5
        ),
        # MLP is kept out by default (tends to need careful tuning on tabular); uncomment if needed.
        # "MLP": MLPClassifier(hidden_layer_sizes=(256,128), random_state=rs, max_iter=500),
    }

    if xgb is not None:
        models["XGBoost"] = xgb.XGBClassifier(
            eval_metric="logloss", random_state=rs, n_jobs=-1,
            n_estimators=4000, learning_rate=0.02,
            max_depth=6, subsample=0.8, colsample_bytree=0.8,
            tree_method="hist",
            early_stopping_rounds=200
        )

    if lgb is not None:
        models["LightGBM"] = lgb.LGBMClassifier(
            random_state=rs, n_jobs=-1, objective="binary",
            n_estimators=3000, learning_rate=0.02,
            num_leaves=64, subsample=0.8, colsample_bytree=0.8
        )
    return models

# -------------- Helpers --------------
def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def safe_name(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_")

def save_text(text: str, path: Path):
    path.write_text(text, encoding="utf-8")

def save_json(obj: dict, path: Path):
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

def plot_confusion(y_true, y_pred, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_roc_from_estimator(estimator, X_test, y_test, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_estimator(estimator, X_test, y_test, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def score_vector(estimator, X):
    if hasattr(estimator, "predict_proba"):
        p = estimator.predict_proba(X)
        if isinstance(p, list):
            p = p[0]
        if getattr(p, "ndim", 1) == 2 and p.shape[1] == 2:
            return p[:, 1]
    if hasattr(estimator, "decision_function"):
        s = estimator.decision_function(X)
        try:
            return s.ravel()
        except Exception:
            return s
    return estimator.predict(X)

def unwrap_calibrated(estimator):
    """
    Get the underlying base estimator from CalibratedClassifierCV.
    New sklearn versions expose `.estimator`; older ones used `.base_estimator`.
    """
    if isinstance(estimator, CalibratedClassifierCV):
        if hasattr(estimator, "calibrated_classifiers_") and estimator.calibrated_classifiers_:
            inner = estimator.calibrated_classifiers_[0]
            if hasattr(inner, "estimator") and inner.estimator is not None:
                return inner.estimator
            if hasattr(inner, "base_estimator") and inner.base_estimator is not None:
                return inner.base_estimator
    return estimator

def feature_importance(estimator, feature_names):
    """
    Return (names, values) if supported (tree-based feature_importances_ or linear coef_).
    Handles calibrated wrappers.
    """
    src = unwrap_calibrated(estimator)

    if hasattr(src, "feature_importances_"):
        return feature_names, np.asarray(src.feature_importances_, dtype=float)

    if hasattr(src, "coef_"):
        coefs = np.asarray(src.coef_, dtype=float)
        if coefs.ndim == 2 and coefs.shape[0] == 1:
            vals = np.abs(coefs[0])
        else:
            vals = np.mean(np.abs(coefs), axis=0)
        return feature_names, vals

    return None

# -------------- Main --------------
def main():
    cfg = CONFIG
    data_csv = Path(cfg["data_csv"])
    models_root = ensure_dir(Path(cfg["models_dir"]))
    results_root = ensure_dir(Path(cfg["results_dir"]))
    global_dir = ensure_dir(models_root / "_global")
    rs = int(cfg["random_state"])

    # Load
    print(f"[INFO] Loading {data_csv} ...")
    df = pd.read_csv(data_csv)
    assert cfg["label_col"] in df.columns, f"Missing label column: {cfg['label_col']}"

    # Select ONLY numeric columns for features (exclude strings like 'url')
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if cfg["label_col"] not in numeric_cols:
        raise ValueError(f"Label column '{cfg['label_col']}' must be numeric (0/1).")
    feature_cols = [c for c in numeric_cols if c != cfg["label_col"]]

    dropped = [c for c in df.columns if c not in feature_cols + [cfg["label_col"]]]
    if dropped:
        print(f"[INFO] Excluding non-numeric columns from X: {dropped}")

    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[cfg["label_col"]].to_numpy(dtype=int)
    assert set(np.unique(y)) <= {0, 1}, "This script expects binary labels {0,1}."

    # Global scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    joblib.dump(scaler, global_dir / "scaler.joblib")
    save_json({"feature_columns": feature_cols}, global_dir / "feature_columns.json")
    print(f"[INFO] Saved global scaler and feature columns to {global_dir}")

    # 70 / 10 / 20 split
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=(1.0 - cfg["train_ratio"]), random_state=rs, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp,
        test_size=(cfg["test_ratio"] / (cfg["val_ratio"] + cfg["test_ratio"])),
        random_state=rs, stratify=y_tmp
    )

    print(f"[INFO] Shapes -> train {X_train.shape}, val {X_val.shape}, test {X_test.shape}")
    cls_w = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
    print(f"[INFO] Class weights (0,1): {cls_w}")

    # Models
    models = build_models(rs)

    # Train/Eval loop
    for name, model in models.items():
        print(f"\n===== Training: {name} =====")
        mname = safe_name(name)
        mdir = ensure_dir(models_root / mname)
        rdir = ensure_dir(results_root / mname)

        # Fit (with val for early stopping if supported)
        if (lgb is not None) and isinstance(model, lgb.LGBMClassifier):
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
            )
        elif (xgb is not None) and isinstance(model, xgb.XGBClassifier):
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            model.fit(X_train, y_train)

        # Save model
        joblib.dump(model, mdir / "model.joblib")

        # Predict & metrics
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, score_vector(model, X_test))
        except Exception:
            auc = None

        metrics = {
            "model": name,
            "samples_evaluated": int(X_test.shape[0]),
            "accuracy": round(float(acc), 6),
            "precision": round(float(prec), 6),
            "recall": round(float(rec), 6),
            "f1_score": round(float(f1), 6),
            "auc_roc": (round(float(auc), 6) if auc is not None else None),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        save_json(metrics, rdir / "metrics.json")

        # Text report
        report = classification_report(y_test, y_pred, digits=4)
        header = f"Performance Analysis for: {name}\n" + "=" * (24 + len(name)) + "\n"
        body = (
            f"Samples (evaluated):\t{X_test.shape[0]}\n"
            f"Accuracy:\t\t{metrics['accuracy']}\n"
            f"Precision:\t\t{metrics['precision']}\n"
            f"Recall:\t\t\t{metrics['recall']}\n"
            f"F1-Score:\t\t{metrics['f1_score']}\n"
            f"AUC-ROC Score:\t\t{metrics['auc_roc']}\n"
            + "=" * (24 + len(name)) + "\n\n"
            "Detailed Classification Report\n"
            "------------------------------\n"
            + report
        )
        save_text(body, rdir / "classification_report.txt")

        # Confusion matrix
        plot_confusion(y_test, y_pred, rdir / "confusion_matrix.png", f"Confusion Matrix: {name}")

        # ROC curve (skip if not supported)
        try:
            plot_roc_from_estimator(model, X_test, y_test, rdir / "roc_curve.png", f"ROC Curve: {name}")
        except Exception:
            pass

        # Feature importance (all numeric features)
        fi = feature_importance(model, np.array(feature_cols))
        if fi is not None:
            fnames, fvals = fi
            df_fi = pd.DataFrame({"feature": fnames, "importance": fvals}).sort_values(
                "importance", ascending=False
            )
            df_fi.to_csv(rdir / "feature_importance.csv", index=False)

            fig_h = max(4.0, 0.36 * len(df_fi))
            fig_w = max(7.0, 0.28 * len(df_fi) if len(df_fi) > 35 else 7.0)
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            ax.barh(df_fi["feature"][::-1], df_fi["importance"][::-1])
            ax.set_title(f"Feature Importances: {name}")
            ax.set_xlabel("Importance")
            fig.tight_layout()
            fig.savefig(rdir / "feature_importance.png", dpi=200)
            plt.close(fig)

        print(f"[OK] Saved model -> {mdir}")
        print(f"[OK] Saved results -> {rdir}")

    print("\n[DONE] All models trained and artifacts saved.")


if __name__ == "__main__":
    main()
