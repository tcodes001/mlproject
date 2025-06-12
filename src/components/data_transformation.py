import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_obj(self): #responsible for data transformation
        try:
            numeric_columns = ['reading_writing_avg']
            categorical_columns=['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            num_pipeline = Pipeline( 
                steps =[ 
                    ("imputer", SimpleImputer(strategy='median')),
                    ("Scaler", StandardScaler())]
            )
            
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='most_frequent')),
                    ("one_hot_encoder", OneHotEncoder(drop='first', sparse_output=False)),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )   
            
            logging.info(f'Categorical columns: {categorical_columns}')
            logging.info(f'Numeric columns: {numeric_columns}')

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numeric_columns),
                    ("categorical_pipeline", cat_pipeline, categorical_columns)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            train_df['reading_writing_avg'] = (train_df['reading_score'] + train_df['writing_score']) / 2
            test_df['reading_writing_avg'] = (test_df['reading_score'] + test_df['writing_score']) / 2

            train_df.drop(columns=['reading_score', 'writing_score'], inplace=True)
            test_df.drop(columns=['reading_score', 'writing_score'], inplace=True)
                        
            logging.info("Reading train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_obj()

            target_column_name = "math_score"
            numeric_columns = ['reading_writing_avg']

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on train and test data")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr= np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object")

            save_object(file_path= self.data_transformation_config.preprocessor_obj_file_path, obj=preprocessing_obj )


            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e,sys)
