from src.coffee_roast_ai.pipeline import TrainingPipeline
from src.coffee_roast_ai.logger import logging

if __name__ == "__main__":
    logging.info("Starting Coffee Roast Classification Training Pipeline")
    pipeline = TrainingPipeline()
    pipeline.run()
    logging.info("Pipeline execution complete.")
