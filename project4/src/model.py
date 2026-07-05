from pathlib import Path
from typing import Any, Dict

from joblib import dump, load
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


def train_model(X, y):
    model = LogisticRegression(solver="liblinear", max_iter=1000)
    model.fit(X, y)
    return model


def evaluate_model(model, X, y) -> Dict[str, Any]:
    """
    Evaluate model performance using multiple metrics.
    """

    # Predictions
    predictions = model.predict(X)

    # Core Metrics
    accuracy = accuracy_score(y, predictions)
    precision = precision_score(y, predictions)
    recall = recall_score(y, predictions)
    f1 = f1_score(y, predictions)

    # Confusion Matrix
    cm = confusion_matrix(y, predictions)

    # Classification Report
    report = classification_report(y, predictions, digits=4)

    # Print results (useful for logs)
    print("\n=== Model Evaluation ===")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(report)

    # Optional: Confusion Matrix Visualization
    if VISUALIZATION_AVAILABLE:
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Negative', 'Positive'], 
                   yticklabels=['Negative', 'Positive'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "classification_report": report
    }


def save_object(object_to_save: Any, path: str):
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    dump(object_to_save, str(path_obj))


def load_object(path: str) -> Any:
    if not Path(path).is_file():
        raise FileNotFoundError(f"Saved object not found: {path}")
    return load(str(path))


def predict_sentiment(model, X):
    return model.predict(X)