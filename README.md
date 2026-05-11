# Bank Chequebook Eligibility

Predicts whether a retail-banking client is eligible to receive a chequebook, from demographic, employment, and account-balance features. Compares Random Forest, Logistic Regression, SVC, and XGBoost on accuracy, precision, recall, F1, and AUC. Built during my end-of-studies internship at UBCI Bank (Tunis, Feb-May 2025).

## Important: data is not included

The notebook expects `data/Base.xlsx` - UBCI's proprietary client data, **not committed** and not shareable. The repo is published as a code artifact only; cells will fail at `pd.read_excel(...)` without the file.

## Tech stack

- Python 3.10+
- pandas (+ `openpyxl`), NumPy
- scikit-learn, XGBoost
- matplotlib, seaborn, missingno, joblib

## Folder structure

```
bank-chequebook-eligibility/
├── notebooks/
│   └── chequebook_eligibility.ipynb
├── src/
│   └── chequebook_eligibility.py
├── requirements.txt
├── .gitignore   # data/ and *.xlsx blocked from commits
└── README.md
```

## How to run

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate      # macOS / Linux
pip install -r requirements.txt
# place Base.xlsx in ./data/
jupyter notebook notebooks/chequebook_eligibility.ipynb
```

## What the code does

Reads `data/Base.xlsx` (column names in French, kept as-is since they're the source schema), derives the target `cheque_eligible = 1 if Défaut == 0 else 0`, and engineers `age` and `seniority` from the customer's birth date and relationship-start date. Label-encodes object columns, fills NaNs with 0, scales with `StandardScaler`, and splits 80/20. Trains Random Forest, Logistic Regression, SVC, KNN, Decision Tree, and XGBoost; for each model it reports a confusion matrix and accuracy / precision / recall / F1 / AUC, saves the confusion-matrix PNG, and serializes the trained model with `joblib`. Uses `ShuffleSplit(n_splits=100)` for cross-validated scores and produces summary bar charts comparing the four main models.
