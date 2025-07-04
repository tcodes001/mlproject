import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
import dill
import pickle
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    
    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_models(X_train, y_train, X_test, y_test,models, param):
    try:
        report={}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            model_name = list(models.keys())[i]
            para = param[model_name]

            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train) 

            model.set_params(**gs.best_params_)

            model.fit(X_train,y_train) # training

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            r2_train_model_score = r2_score(y_train, y_train_pred)
            r2_test_score = r2_score(y_test, y_test_pred)
            mae_test_score = mean_absolute_error(y_test, y_test_pred)
            rmse_test_score = np.sqrt(mean_squared_error(y_test, y_test_pred))

            report[model_name] = { "R2" : r2_test_score, 
                                   "RMSE" : rmse_test_score,
                                   "MAE" : mae_test_score }

        return report
    
    except Exception as e:
        raise CustomException(e,sys)
    

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    
    except Exception as e:
        raise CustomException(e,sys)
    
    
def get_background_data(train_path='artifacts/train.csv'):
    df = pd.read_csv(train_path)

    # Safely merge and drop old features
    if 'reading_writing_avg' not in df.columns:
        df['reading_writing_avg'] = (df['reading_score'] + df['writing_score']) / 2
        df.drop(columns=['reading_score', 'writing_score'], inplace=True)

    if 'math_score' in df.columns:
        df.drop(columns=['math_score'], inplace=True)

    return df
