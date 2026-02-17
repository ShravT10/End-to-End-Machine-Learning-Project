from src.exception import CustomException
import os
import sys 
from dataclasses import dataclass

import numpy as np
import pandas as pd 
from sklearn.compose import ColumnTransformer 
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder , StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


class DataTranformationConfig:
    preprocessor_obj_file = os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTranformationConfig()
    
    def get_data_transformer_object(self):

        '''
        Basically Create A Preprocessor Object and return it
        '''
        try:
            numerical_feature = ["writing_score","reading_score"]
            categorical_feature = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            numerical_pipeline = Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder',OneHotEncoder(sparse_output=False)),
                    ('StandardScaler',StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",numerical_pipeline,numerical_feature),
                    ("cat_pipeline",categorical_pipeline,categorical_feature)
                ]
            )

            logging.info("preprocessor ready")

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transfromation(self,train_path,test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("read train and test data complete")
            logging.info("obtaining preprocessing object")

            preprocess_obj = self.get_data_transformer_object()

            target_var = "math_score"
            numerical_feature = ["writing_score","reading_score"]

            input_train_df = train_df.drop([target_var],axis=1)
            target_var_train = train_df[target_var]

            input_test_df = test_df.drop([target_var],axis=1)
            target_var_test = test_df[target_var]

            logging.info("Applying Preprocessor on Train and Test data")

            train_df = preprocess_obj.fit_transform(input_train_df)
            test_df = preprocess_obj.transform(input_test_df)

            train_arr = np.c_[
                train_df , np.array(target_var_train),
            ]

            test_arr = np.c_[
                test_df , np.array(target_var_test)
            ]

            logging.info("preprocessing complete")

            save_object(file_path=self.data_transformation_config.preprocessor_obj_file,
            obj=preprocess_obj)

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file
            )




        except Exception as e:
            raise CustomException(e,sys)
