import os
import mlflow
import mlflow.keras

from ..data_ingest import download_and_list_files
from ..data_loader import CoffeeDataLoader
from ..model_engine import CoffeeModelEngine
from ..utils import read_params
from ..logger import logging


class TrainingPipeline:
    """
    Orchestrates the full ML workflow in four stages:
      1. Data Ingestion  — download dataset from Kaggle
      2. Data Loading    — build train / val / test generators
      3. Model Training  — train InceptionV3 with MLflow experiment tracking
      4. Model Saving    — persist best weights to disk
    """

    def __init__(self):
        self.config = read_params()
        mlflow.set_experiment("coffee-roast-classification")

    def run(self, model_output_path: str = "models/inception_v1.hdf5"):
        # ── Stage 1: Data Ingestion ─────────────────────────────────────────
        logging.info("=== Stage 1/4: Data Ingestion ===")
        dataset_path = download_and_list_files()

        # ── Stage 2: Data Loading ───────────────────────────────────────────
        logging.info("=== Stage 2/4: Data Loading ===")
        loader = CoffeeDataLoader(
            train_dir=f"{dataset_path}/train/",
            test_dir=f"{dataset_path}/test/"
        )
        train_ds, val_ds = loader.get_train_val_loaders()
        test_ds = loader.get_test_loader()

        # ── Stage 3: Training with MLflow Tracking ──────────────────────────
        logging.info("=== Stage 3/4: Model Training (MLflow run started) ===")
        engine = CoffeeModelEngine()
        model = engine.build_inception_model()

        with mlflow.start_run(run_name="inception_v3_training") as run:

            # Log all hyperparameters from params.yaml
            mlflow.log_params({
                "base_model": self.config["model"]["base_model"],
                "learning_rate": self.config["model"]["learning_rate"],
                "dropout_rate": self.config["model"]["dropout_rate"],
                "dense_units": self.config["model"]["dense_units"],
                "epochs": self.config["model"]["epochs"],
                "batch_size": self.config["data"]["batch_size"],
                "image_size": str(self.config["data"]["image_size"]),
                "aug_rotation": self.config["augmentation"]["rotation_range"],
                "aug_zoom": self.config["augmentation"]["zoom_range"],
            })

            # Train
            history = model.fit(
                train_ds,
                epochs=self.config["model"]["epochs"],
                validation_data=val_ds
            )

            # Log per-epoch metrics so MLflow UI shows training curves
            acc_key = "categorical_accuracy"
            val_acc_key = "val_categorical_accuracy"
            for epoch, (acc, val_acc, loss, val_loss) in enumerate(zip(
                history.history[acc_key],
                history.history[val_acc_key],
                history.history["loss"],
                history.history["val_loss"]
            )):
                mlflow.log_metrics(
                    {
                        "train_accuracy": round(float(acc), 4),
                        "val_accuracy": round(float(val_acc), 4),
                        "train_loss": round(float(loss), 4),
                        "val_loss": round(float(val_loss), 4),
                    },
                    step=epoch + 1
                )

            # Final evaluation on held-out test set
            test_loss, test_acc = model.evaluate(test_ds)
            mlflow.log_metrics({
                "test_accuracy": round(float(test_acc), 4),
                "test_loss": round(float(test_loss), 4),
            })

            # Persist the trained Keras model as an MLflow artifact
            mlflow.keras.log_model(model, artifact_path="model")

            logging.info(f"MLflow Run ID   : {run.info.run_id}")
            logging.info(f"Test Accuracy   : {test_acc:.4f}")
            logging.info(f"Test Loss       : {test_loss:.4f}")

        # ── Stage 4: Save Model ─────────────────────────────────────────────
        logging.info("=== Stage 4/4: Saving Model ===")
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        engine.save_model(model_output_path)
        logging.info(f"Pipeline complete. Model saved → {model_output_path}")

        return history
