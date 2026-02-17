import os 
import sys 
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor
)
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object , evaluate_models

@dataclass
class ModelTrainerConfig:
    train_model_filepath = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info("Splitting Train & Test")
            X_train , y_train , X_test , y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Linear Regression" : LinearRegression(),
                "K Neighbor" : KNeighborsRegressor(),
                "Decision Tree" : DecisionTreeRegressor(),
                "Random Forest" : RandomForestRegressor(),
                "Gradient Boosting" : GradientBoostingRegressor(),
                "Adaboost" : AdaBoostRegressor(),
                "XGboost" : XGBRegressor(),
                "Cat Boost" : CatBoostRegressor(verbose = False)
            } 

            models_report:dict = evaluate_models(X_train=X_train,y_train=y_train,
                                X_test=X_test,y_test=y_test , models = models)
            
            # To get best model score from dict
            best_model_score = max(sorted(models_report.values()))

            # To get best model name from dict
            best_model_name = list(models_report.keys())[
                list(models_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.60:
                raise CustomException("No best model found")

            logging.info(f"Best found model on both training and testing dataset: {best_model_name}")

            save_object(file_path=self.model_trainer_config.train_model_filepath,
            obj=best_model)

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test,predicted)

            return r2_square
        
        except Exception as e:
            raise CustomException(e,sys) 