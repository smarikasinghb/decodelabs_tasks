"""
Project 2: Data Classification Using AI
Decode Labs - AI Internship 2026

Pipeline: Iris Dataset → Feature Scaling → KNN → Confusion Matrix + F1 Score
Algorithm: K-Nearest Neighbors (K=5)
"""

# ── IMPORTS ──────────────────────────────────────────────
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ── PHASE 1: INPUT — Load & Understand the Data ──────────
iris = load_iris()
X = iris.data      # Features: sepal length, sepal width, petal length, petal width
y = iris.target    # Labels: 0=Setosa, 1=Versicolor, 2=Virginica

print("=" * 50)
print("PHASE 1: DATASET LOADED")
print("=" * 50)
print(f"Total samples  : {X.shape[0]}")
print(f"Features       : {X.shape[1]}  {iris.feature_names}")
print(f"Classes        : {list(iris.target_names)}")

# ── PHASE 2: PROCESS — Scale, Split, Train ───────────────

# Step 1: Feature Scaling (StandardScaler: Mean=0, Variance=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Train-Test Split (80% train, 20% test, shuffled)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

print(f"\nTraining samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")

# Step 3: Train KNN Model (K=5)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)          # FIT: memorize the map
predictions = model.predict(X_test)  # PREDICT: apply logic

print("\n" + "=" * 50)
print("PHASE 2: MODEL TRAINED (KNN, K=5)")
print("=" * 50)

# ── PHASE 3: OUTPUT — Evaluate ───────────────────────────
print("\n" + "=" * 50)
print("PHASE 3: OUTPUT VALIDATION")
print("=" * 50)

# Confusion Matrix
cm = confusion_matrix(y_test, predictions)
print("\nConfusion Matrix:")
print(f"{'':15} Predicted Setosa  Versicolor  Virginica")
for i, row in enumerate(cm):
    print(f"Actual {iris.target_names[i]:10} {row}")

# Classification Report (Precision, Recall, F1)
print("\nClassification Report:")
print(classification_report(y_test, predictions, target_names=iris.target_names))

# ── BONUS: Predict a single new flower ───────────────────
print("=" * 50)
print("BONUS: SINGLE PREDICTION")
print("=" * 50)
sample = [[5.1, 3.5, 1.4, 0.2]]  # Known Setosa measurements
sample_scaled = scaler.transform(sample)
result = model.predict(sample_scaled)
print(f"Input features : {sample[0]}")
print(f"Predicted class: {iris.target_names[result[0]]}")
