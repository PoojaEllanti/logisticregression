import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.metrics import ( # type: ignore
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset from Kaggle CSV
df = pd.read_csv("data.csv")

# 2. Drop unused columns
df.drop(["id", "Unnamed: 32"], axis=1, inplace=True)

# 3. Convert target 'diagnosis' to binary (M=1, B=0)
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

# 4. Features and target split
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Train logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 8. Predict
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# 9. Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_proba):.2f}")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# 10. Sigmoid Function Plot
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = np.linspace(-10, 10, 100)
sig = sigmoid(z)
plt.plot(z, sig)
plt.title("Sigmoid Function")
plt.xlabel("z")
plt.ylabel("Sigmoid(z)")
plt.grid()
plt.show()

# 11. Try custom threshold
custom_thresh = 0.3
y_custom_pred = (y_proba >= custom_thresh).astype(int)
print(f"\nClassification Report (Threshold={custom_thresh}):\n",
      classification_report(y_test, y_custom_pred))
