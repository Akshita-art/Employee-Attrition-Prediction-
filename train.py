"""Train and compare 5 classifiers for employee attrition prediction."""
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42

df = pd.read_csv("data/employee_attrition_dataset.csv")
df = df.drop(columns=["Employee_ID"])
df["Attrition"] = (df["Attrition"] == "Yes").astype(int)

X, y = df.drop(columns="Attrition"), df["Attrition"]
cat_cols = X.select_dtypes(exclude="number").columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)


def make_pipe(clf):
    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    return Pipeline([("pre", pre), ("clf", clf)])


# class_weight="balanced" because only ~19% of employees left (KNN has no such option)
models = {
    "Logistic Regression": (
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        {"clf__C": [0.01, 0.1, 1, 10]}),
    "SVM": (
        SVC(probability=True, class_weight="balanced", random_state=RANDOM_STATE),
        {"clf__C": [0.1, 1, 10], "clf__kernel": ["linear", "rbf"]}),
    "KNN": (
        KNeighborsClassifier(),
        {"clf__n_neighbors": [3, 5, 7, 11, 15], "clf__weights": ["uniform", "distance"]}),
    "Decision Tree": (
        DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        {"clf__max_depth": [3, 5, 8, None], "clf__min_samples_leaf": [1, 5, 10]}),
    "Random Forest": (
        RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        {"clf__n_estimators": [200, 400], "clf__max_depth": [5, 10, None],
         "clf__min_samples_leaf": [1, 3]}),
}

cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
results, fitted = [], {}
for name, (clf, grid) in models.items():
    gs = GridSearchCV(make_pipe(clf), grid, cv=cv, scoring="roc_auc", n_jobs=-1)
    gs.fit(X_train, y_train)
    best = gs.best_estimator_
    pred = best.predict(X_test)
    proba = best.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    r = {
        "model": name,
        "best_params": {k.replace("clf__", ""): v for k, v in gs.best_params_.items()},
        "cv_roc_auc": round(gs.best_score_, 4),
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    results.append(r)
    fitted[name] = best
    print(f"{name:20s} cvAUC={r['cv_roc_auc']:.3f} acc={r['accuracy']:.3f} "
          f"prec={r['precision']:.3f} rec={r['recall']:.3f} "
          f"f1={r['f1']:.3f} auc={r['roc_auc']:.3f}")

# Pick the winner on cross-validated ROC-AUC (training data only) to avoid test-set leakage.
winner = max(results, key=lambda r: r["cv_roc_auc"])["model"]
print("Selected model:", winner)

# For deployment, refit the chosen pipeline (same hyperparameters) on all rows.
final = fitted[winner]
final.fit(X, y)
joblib.dump(final, "model/best_model.joblib")

meta = {
    "selected_model": winner,
    "results": results,
    "positive_rate": round(float(y.mean()), 4),
    "n_rows": int(len(df)),
    "numeric_features": num_cols,
    "categorical_features": {c: sorted(X[c].unique().tolist()) for c in cat_cols},
    "numeric_ranges": {c: [int(X[c].min()), int(X[c].max())] for c in num_cols},
}
with open("model/metrics.json", "w") as f:
    json.dump(meta, f, indent=2)
