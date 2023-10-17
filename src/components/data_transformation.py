import os 
import sys   
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

from src.utils import save_object



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            columns = [
                "cap-shape",
                "cap-surface",
                "cap-color",
                "bruises",
                "odor",
                "gill-attachment",
                "gill-spacing",
                "gill-size",
                "gill-color",
                "stalk-shape",
                "stalk-root",
                "stalk-surface-above-ring",
                "stalk-surface-below-ring",
                "stalk-color-above-ring",
                "stalk-color-below-ring",
                "veil-type",
                "veil-color",
                "ring-number",
                "ring-type",
                "spore-print-color",
                "population",
                "habitat"
            ]

            col_pipeline = Pipeline(
                steps=[
                    ("one_hot_encoding", OneHotEncoder(sparse_output=False)),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Columns: {columns}")

            preprocessor=ColumnTransformer(
                [
                    ("columns", col_pipeline, columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)

    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and Test data have been read")

            logging.info("Obtaining Preprocessing object...")
            preprocessing_obj = self.get_data_transformer_object()

            target_column = "class"
            input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_df[target_column]
            
            logging.info(f"Applying preprocessing object on training and testing dataframe.")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saving preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
            