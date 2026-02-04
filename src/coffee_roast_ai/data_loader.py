import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from CoffeeBeanRoastClassification.src.coffee_roast_ai import data_ingest
from .utils import read_params
from pathlib import Path
from .data_ingest import download_and_list_files
    
class CoffeeDataLoader:
    def __init__(self, train_dir, test_dir):
        """
        Initializes the data loader using the logic from your Inception notebook.
        """
        self.config = read_params()
        self.train_dir = Path(train_dir)
        self.test_dir = Path(test_dir)
        self.image_size = self.config['data']['image_size']
        self.batch_size = self.config['data']['batch_size']
        
        # We use the exact augmentation parameters from your Cell 9
        self.datagen = ImageDataGenerator(
            rescale=self.config['augmentation']['rescale'],
            rotation_range=self.config['augmentation']['rotation_range'],
            width_shift_range=self.config['augmentation']['width_shift_range'],
            height_shift_range=self.config['augmentation']['height_shift_range'],
            shear_range=self.config['augmentation']['shear_range'],
            zoom_range=self.config['augmentation']['zoom_range'],
            horizontal_flip=self.config['augmentation']['horizontal_flip'],
            vertical_flip=self.config['augmentation']['vertical_flip'],
            fill_mode=self.config['augmentation']['fill_mode'],
            validation_split=self.config['data']['validation_split'] # Using 10% for validation as per your notebook
        )

    def get_train_val_loaders(self):
        """
        Returns the training and validation generators.
        """
        train_ds = self.datagen.flow_from_directory(
            self.train_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            color_mode="rgb",
            shuffle=True
        )

        val_ds = self.datagen.flow_from_directory(
            self.train_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            color_mode="rgb",
            shuffle=True
        )
        
        return train_ds, val_ds

    def get_test_loader(self):
        """
        Returns the test generator (no shuffle for evaluation).
        """
        return self.datagen.flow_from_directory(
            self.test_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            color_mode="rgb",
            shuffle=False
        )

# Example Usage (You can put this in your test_pipeline.py)
if __name__ == "__main__":
    # These paths should match your local setup or kagglehub download path
    dataset_path =  download_and_list_files()
    TRAIN_PATH = f"{dataset_path}/train/"
    TEST_PATH = f"{dataset_path}/test/"
    
    loader = CoffeeDataLoader(TRAIN_PATH, TEST_PATH)
    train_gen, val_gen = loader.get_train_val_loaders()
    
    print(f"Classes found: {train_gen.class_indices}")