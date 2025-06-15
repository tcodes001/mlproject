from src.components.data_ingestion import DataIngestion
from src.components.data_ingestion import DataIngestionConfig
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig

if __name__ == "__main__":
    # Step 1: Data Ingestion
    data_ingestion = DataIngestion()
    train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

    # Step 2: Data Transformation
    data_transformation = DataTransformation()
    train_array, test_array, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

    # Step 3: Model Training
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer(train_array, test_array)
