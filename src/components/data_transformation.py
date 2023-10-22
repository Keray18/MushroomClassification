import os 
import sys   
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
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
                "odor", 
                "gill_color", 
                "spore_print_color", 
                "cap_color", 
                "bruises", 
                "stalk_surface_above_ring", 
                "stalk_surface_below_ring", 
                "gill_size", 
                "ring_type", 
                "population"
            ]

            col_pipeline = Pipeline(
                steps=[
                    ("one_hot_encoding", OneHotEncoder(sparse_output=False)),
                    ("pca", PCA(n_components=2))
                    
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

            # Preprocessing Target Variables
            target_feature_train_df = target_feature_train_df.map({ "e": 0, "p": 1})
            target_feature_test_df = target_feature_test_df.map({ "e": 0, "p": 1})
            logging.info("Target Variables have been preprocessed.")
            
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
            