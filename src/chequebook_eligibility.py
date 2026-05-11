"""Chequebook eligibility prediction (UBCI Bank internship, Feb-May 2025).

Classifies bank clients as eligible or not for a chequebook based on demographic
and account-balance features. Compares Random Forest, Logistic Regression, SVC,
and XGBoost; reports confusion matrices, ROC curves, and summary metrics.

IMPORTANT: this script expects a private bank dataset (data/Base.xlsx) that is
NOT included in this repository. The code will not run without it.
"""


# %%
#  Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#  Load your Excel dataset
dataset = pd.read_excel('data/Base.xlsx  # private bank dataset - not included in this repo')

#  Confirm structure
print(dataset.columns)
dataset.head()

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_excel("data/Base.xlsx  # private bank dataset - not included in this repo")

# Create binary target
df['cheque_eligible'] = df['Défaut'].apply(lambda x: 1 if x == 0 else 0)

# Handle age and seniority
df['age'] = pd.to_datetime('today').year - pd.to_datetime(df["DATE NAISSANCE                -TIER"], errors='coerce').dt.year
df['seniority'] = pd.to_datetime('today').year - pd.to_datetime(df["DATE ENTREE EN RELATION TIERS -TIER"], errors='coerce').dt.year

# Select features
features = [
    'Filière', 'Libellé Sexe', 'Libellé Situation Familiale',
    'NBR ENFANTS                   -TIER', 'age', 'seniority',
    'Secteur d\'activité PC', 'Libellé CSP', 'Code CSP',
    'rating y', 'rating y-1',
    'Moy Solde_en_fin_de_mois',
    'Moy Solde_en_fin_de_mois (Semestre-1)',
    'Moy Solde_en_fin_de_mois (Année-1)'
]
X = df[features].copy()
y = df['cheque_eligible']

# Encode categorical variables
for col in X.select_dtypes(include='object').columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# Handle missing values
X = X.fillna(0)

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

# Load and prepare dataset
df = pd.read_excel("data/Base.xlsx  # private bank dataset - not included in this repo")
df.columns = df.columns.str.strip()

df['cheque_eligible'] = df['Défaut'].apply(lambda x: 1 if x == 0 else 0)
df['age'] = pd.to_datetime('today').year - pd.to_datetime(df["DATE NAISSANCE                -TIER"], errors='coerce').dt.year
df['seniority'] = pd.to_datetime('today').year - pd.to_datetime(df["DATE ENTREE EN RELATION TIERS -TIER"], errors='coerce').dt.year

features = [
    'Filière', 'Libellé Sexe', 'Libellé Situation Familiale',
    'NBR ENFANTS                   -TIER', 'age', 'seniority',
    "Secteur d'activité PC", 'Libellé CSP', 'Code CSP',
    'rating y', 'rating y-1',
    'Moy Solde_en_fin_de_mois',
    'Moy Solde_en_fin_de_mois (Semestre-1)',
    'Moy Solde_en_fin_de_mois (Année-1)'
]

X = df[features].copy()
y = df['cheque_eligible']

for col in X.select_dtypes(include='object').columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

X = X.fillna(0)
X_scaled = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

output_dir = "outputs/models_chequebook"
os.makedirs(output_dir, exist_ok=True)

# %%
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("📊 Random Forest Metrics")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_prob))
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Not Eligible", "Eligible"], yticklabels=["Not Eligible", "Eligible"])
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(f"{output_dir}/RandomForest_confusion_matrix.png")
plt.close()

# Save model
joblib.dump(model, f"{output_dir}/RandomForest_chequebook_model.pkl")

# %%
from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve - Random Forest')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig(f"{output_dir}/RandomForest_ROC_Curve.png")
plt.close()

# %%
sns.countplot(x='Défaut', data=df, palette=["#126b4a", "#D9534F"])

# %%
import missingno as msno
import matplotlib.pyplot as plt

msno.matrix(df)
plt.title("Missing-value visualization - Customer data")
plt.show()

# %%
msno.bar(df)
plt.title("Non-missing values per column - Customer data")
plt.show()

# %%
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='cheque_eligible', data=df, palette=["#D9534F", "#126b4a"])
plt.title('Client distribution by chequebook eligibility')
plt.xlabel('Eligibility (0 = Not eligible, 1 = Eligible)')
plt.ylabel('Number of clients')
plt.show()

# %%
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("📊 Logistic Regression Metrics")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_prob))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=["Not Eligible", "Eligible"], yticklabels=["Not Eligible", "Eligible"])
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(f"{output_dir}/LogisticRegression_confusion_matrix.png")
plt.close()

joblib.dump(model, f"{output_dir}/LogisticRegression_chequebook_model.pkl")

# %%
from sklearn.svm import SVC

model = SVC(probability=True)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("📊 SVC Metrics")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_prob))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=["Not Eligible", "Eligible"], yticklabels=["Not Eligible", "Eligible"])
plt.title("Confusion Matrix - SVC")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(f"{output_dir}/SVC_confusion_matrix.png")
plt.close()

joblib.dump(model, f"{output_dir}/SVC_chequebook_model.pkl")

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, classification_report
)
from sklearn.model_selection import cross_val_score, ShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns

# Dictionary of models to test
classifiers = {
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    "Logistic Regression": LogisticRegression(solver='liblinear'),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Classifier": SVC(probability=True),
    "Decision Tree": DecisionTreeClassifier()
    # XGBoost can be added here if enabled
}

# Cross-validation
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=42)

for name, clf in classifiers.items():
    print(f"\n=== {name} ===")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Scores
    acc = accuracy_score(y_test, y_pred)
    pr = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Normalized rates
    total = tn + fp + fn + tp
    print(f"Confusion Matrix:\n{cm}")
    print(f"Taux : TN: {tn/total:.2f}, FP: {fp/total:.2f}, FN: {fn/total:.2f}, TP: {tp/total:.2f}")
    print(f"Specificity (TN rate): {tn / (tn + fp):.2f}")
    print(f"Sensitivity (Recall): {tp / (tp + fn):.2f}")

    # Overall metrics
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {pr:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(classification_report(y_test, y_pred))

    # Confusion-matrix visualization
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Not Eligible", "Eligible"],
                yticklabels=["Not Eligible", "Eligible"])
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # Optional: training + learning curve
    scores = cross_val_score(clf, X_train, y_train, cv=cv)
    print(f"Cross-validated score (mean): {scores.mean():.4f}")

# %%
from xgboost import XGBClassifier

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("📊 XGBoost Metrics")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_prob))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=["Not Eligible", "Eligible"], yticklabels=["Not Eligible", "Eligible"])
plt.title("Confusion Matrix - XGBoost")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(f"{output_dir}/XGBoost_confusion_matrix.png")
plt.close()

joblib.dump(model, f"{output_dir}/XGBoost_chequebook_model.pkl")

# %%
import matplotlib.pyplot as plt

# Model names
models = ["Random Forest", "SVC", "XGBoost", "Logistic Regression"]

# Accuracy (training and testing) - adjust to your own results
train_accuracies = [0.91, 0.89, 0.89, 0.88]
test_accuracies =  [0.89, 0.89, 0.88, 0.88]

# Bar width and positioning
bar_width = 0.35
x = range(len(models))

# Build the compact figure
plt.figure(figsize=(6, 4))
bars1 = plt.bar([p - bar_width/2 for p in x], train_accuracies, width=bar_width,
                label='Training', color='#5DADE2')
bars2 = plt.bar([p + bar_width/2 for p in x], test_accuracies, width=bar_width,
                label='Testing', color='#E67E22')

# Add numeric values above each bar
for bar in bars1 + bars2:
    height = bar.get_height()
    plt.annotate(f'{height:.2f}',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 2), textcoords="offset points",
                 ha='center', fontsize=7)

# Minimalist appearance
plt.xticks(x, models, rotation=25, fontsize=7)
plt.yticks(fontsize=7)
plt.ylim(0.85, 0.92)
plt.ylabel('Accuracy', fontsize=8)
plt.legend(fontsize=7)
plt.tight_layout()

# Display the chart
plt.show()

# %%
import matplotlib.pyplot as plt

# Dataset sizes
train_size = len(X_train)
test_size = len(X_test)
total_size = train_size + test_size

# Data for the chart
labels = ['Train (80%)', 'Test (20%)']
sizes = [train_size, test_size]
colors = ['#4CAF50', '#2196F3']  # Green for train, blue for test

# Build the chart
plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, colors=colors, autopct=lambda p: '{:.1f}%\n({:.0f} samples)'.format(p, p * total_size / 100),
        startangle=90, textprops={'fontsize': 12})
plt.title('Data split: Training vs Test', fontsize=14)
plt.axis('equal')  # Perfect circle
plt.tight_layout()
plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np

# Models
models = ['Random Forest', 'Logistic Regression', 'SVC (SVM)', 'XGBoost']

# Metric values
accuracy = [0.9748, 0.9577, 0.9664, 0.9708]
precision = [0.9763, 0.9582, 0.9682, 0.9719]
f1_score = [0.9887, 0.9781, 0.9825, 0.9845]

# Bar positions
x = np.arange(len(models))
bar_width = 0.25

# Build the chart
plt.figure(figsize=(10, 6))
plt.bar(x - bar_width, accuracy, width=bar_width, label='Accuracy', color='green')
plt.bar(x, precision, width=bar_width, label='Precision', color='red')
plt.bar(x + bar_width, f1_score, width=bar_width, label='F1-score', color='blue')

# Add labels and title
plt.xlabel('Models')
plt.ylabel('Score')
plt.title('Performance-metric comparison')
plt.xticks(x, models, rotation=15)
plt.ylim(0.94, 1.0)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Display
plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np

# Models
models = ['Random Forest', 'Logistic Regression', 'SVC (SVM)', 'XGBoost']

# Metric values
accuracy = [0.9748, 0.9577, 0.9664, 0.9708]
precision = [0.9763, 0.9582, 0.9682, 0.9719]
f1_score = [0.9887, 0.9781, 0.9825, 0.9845]

# Bar positions
x = np.arange(len(models))
bar_width = 0.25

# Build the chart
plt.figure(figsize=(10, 6))
plt.bar(x - bar_width, accuracy, width=bar_width, label='Accuracy', color='lightgreen')
plt.bar(x, precision, width=bar_width, label='Precision', color='lightcoral')
plt.bar(x + bar_width, f1_score, width=bar_width, label='F1-score', color='lightskyblue')

# Add labels and title
plt.xlabel('Models')
plt.ylabel('Score')
plt.title('Performance-metric comparison')
plt.xticks(x, models, rotation=0)  # Rotation 0 = horizontal
plt.ylim(0.94, 1.0)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Display
plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np

# Models
models = ['Random Forest', 'Logistic Regression', 'SVC (SVM)', 'XGBoost']

# Metric values
accuracy = [0.9748, 0.9577, 0.9664, 0.9708]
precision = [0.9763, 0.9582, 0.9682, 0.9719]
f1_score = [0.9887, 0.9781, 0.9825, 0.9845]

# Colors extracted from the image
color_accuracy = '#9dc183'   # Dark green
color_precision = '#2e8b57' # Light green / gray
color_f1 = '#004ba0'        # Blue

# Bar positions
x = np.arange(len(models))
bar_width = 0.25

# Build the chart
plt.figure(figsize=(10, 6))
plt.bar(x - bar_width, accuracy, width=bar_width, label='Accuracy', color=color_accuracy)
plt.bar(x, precision, width=bar_width, label='Precision', color=color_precision)
plt.bar(x + bar_width, f1_score, width=bar_width, label='F1-score', color=color_f1)

# Layout
plt.xlabel('Models')
plt.ylabel('Score')
plt.title('Performance-metric comparison')
plt.xticks(x, models, rotation=0)
plt.ylim(0.94, 1.0)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Display
plt.show()
