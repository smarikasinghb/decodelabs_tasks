import logging
from pathlib import Path

from src.feature_engineering import build_features, transform_features
from src.model import evaluate_model, save_object, train_model
from src.prepare_dataset import prepare_dataset


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    configure_logging()
    project_root = Path(__file__).resolve().parent
    train_path = project_root / "dataset" / "train_data.csv"
    test_path = project_root / "dataset" / "test_data.csv"
    model_path = project_root / "models" / "sentiment_model.joblib"
    vectorizer_path = project_root / "models" / "vectorizer.joblib"

    try:
        logging.info("Loading and preparing the dataset...")
        merged_data = prepare_dataset(str(train_path), str(test_path))

        train_data = merged_data[merged_data["dataset_type"] == "train"]
        test_data = merged_data[merged_data["dataset_type"] == "test"]

        if train_data.empty or test_data.empty:
            raise ValueError("Train and test splits must contain at least one sample.")

        logging.info("Building training features...")
        X_train, vectorizer = build_features(train_data)
        y_train = train_data["sentiment"].astype(int)

        logging.info("Training the logistic regression model...")
        model = train_model(X_train, y_train)

        logging.info("Building test features...")
        X_test = transform_features(test_data, vectorizer)
        y_test = test_data["sentiment"].astype(int)

        # ✅ EVALUATION (CORRECT PLACE)
        logging.info("Evaluating model performance...")
        metrics = evaluate_model(model, X_test, y_test)

        logging.info("Test accuracy: %.4f", metrics["accuracy"])
        logging.info("Precision: %.4f", metrics["precision"])
        logging.info("Recall: %.4f", metrics["recall"])
        logging.info("F1 Score: %.4f", metrics["f1"])
        logging.info("Classification report:\n%s", metrics["classification_report"])

        logging.info("Saving the trained model to %s", model_path)
        save_object(model, str(model_path))

        logging.info("Saving the TF-IDF vectorizer to %s", vectorizer_path)
        save_object(vectorizer, str(vectorizer_path))

        logging.info("Pipeline completed successfully.")

    except FileNotFoundError as error:
        logging.error(error)
    except ValueError as error:
        logging.error("Data validation failed: %s", error)
    except Exception as error:
        logging.exception("An unexpected error occurred: %s", error)


if __name__ == "__main__":
    main()