# --- Step 1: Imports and Setup ---
from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix, auc
)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

# --- Step 2: Load Data ---
data = fetch_ucirepo(id=350)
X = data.data.features
y = data.data.targets
target_col = y.columns[0]
df = pd.concat([X, y], axis=1)

# --- Step 3: Exploratory Data Analysis (EDA) ---
print("Dataset Shape:", df.shape)
print("Target Distribution:\n", df[target_col].value_counts(normalize=True))

# Target distribution barplot
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x=target_col)
plt.title("Distribution of Default vs Non-default")
plt.xlabel("Default Status (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.grid()
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(numeric_only=True), cmap='coolwarm', center=0)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# --- Step 4: Preprocessing ---
X_encoded = pd.get_dummies(X, drop_first=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# --- Step 5: Model Training ---
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(n_estimators=100)
}

results = {}

for model_name, model in models.items():
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)
    
    results[model_name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    print(f"\n📌 {model_name} Results:")
    for k, v in results[model_name].items():
        if k != 'confusion_matrix':
            print(f"{k.title()}: {v:.4f}")
    print("Confusion Matrix:\n", results[model_name]['confusion_matrix'])

    # --- Confusion Matrix Plot ---
    cm = results[model_name]['confusion_matrix']
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

# --- Step 6: ROC Curve Visualization ---
plt.figure(figsize=(10, 6))
for model_name, model in models.items():
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid()
plt.tight_layout()
plt.show()

# --- Step 7: Feature Importance (Random Forest) ---
rf_model = models['Random Forest']
importances = rf_model.feature_importances_
feature_names = X_encoded.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.title("Top 15 Feature Importances - Random Forest")
sns.barplot(x=importances[indices][:15], y=feature_names[indices][:15], palette='viridis')
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# --- Step 8: Business Interpretation ---
print("\n🔍 Business Insights:")

print("\n1️⃣ Top Features Driving Default Risk (Random Forest):")
for i in range(10):
    print(f"{i+1}. {feature_names[indices[i]]} ({importances[indices[i]]:.4f})")

print("\n2️⃣ Understanding the Confusion Matrix:")
print("- False Positives (predict default, no default): Lost good customers")
print("- False Negatives (predict good, actually default): Direct financial loss")

print("\n3️⃣ Business Strategy:")
print("- Prioritize reducing **false negatives**: better for institutional risk control.")
print("- Use high-recall models + conservative threshold tuning for loan approvals.")

print("\n✅ All modeling, evaluation, and visualization steps completed.")
