# Employee Attrition Predictor

Flask web app that predicts whether an employee is likely to leave, using a scikit-learn classifier.

Models compared in `train.py`: Logistic Regression, SVM, KNN, Decision Tree, Random Forest
(5-fold CV grid search, stratified 80/20 split, class weighting for the ~19% attrition rate).

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train.py        # trains models, writes model/best_model.joblib + model/metrics.json
python app.py          # http://127.0.0.1:5000
```

## Deploy (Render, free tier)
1. Push this repo to GitHub.
2. On render.com: New > Web Service > connect the repo. `render.yaml` is picked up automatically
   (build: `pip install -r requirements.txt`, start: `gunicorn app:app`).

## Important note on the dataset
All five models score ~0.5 ROC-AUC on the supplied dataset, i.e. no better than random guessing.
No feature shows a meaningful relationship with attrition, so the data appears to be synthetic/random.
The pipeline and website are fully working; swap in real HR data (same columns) and re-run `train.py`.
