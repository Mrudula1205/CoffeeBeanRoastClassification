from src.coffee_roast_ai.data_ingest import download_and_list_files
from src.coffee_roast_ai.data_loader import CoffeeDataLoader
from src.coffee_roast_ai.model_engine import CoffeeModelEngine

def run_pipeline():
    # 1. Get Data Path
    dataset_path = download_and_list_files()
    
    # 2. Setup Loaders
    loader = CoffeeDataLoader(
        train_dir=f"{dataset_path}/train/",
        test_dir=f"{dataset_path}/test/"
    )
    train_ds, val_ds = loader.get_train_val_loaders()
    
    # 3. Build and Train Model
    engine = CoffeeModelEngine()
    model = engine.build_inception_model()
    
    print("Starting training...")
    model.fit(
        train_ds, 
        epochs=engine.model_cfg['epochs'], 
        validation_data=val_ds
    )
    
    # 4. Save
    engine.save_model("models/inception_v1.hdf5")
    print("Pipeline complete!")

if __name__ == "__main__":
    run_pipeline()