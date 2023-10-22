import sys   
import os
import pandas as pd   
from src.exception import CustomException
from src.utils import load_object   


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            print("Before Loading")

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            print("After Loading")
            data_scaled = preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(self,
                odor: str,
                gill_color: str,
                spore_print_color: str,
                cap_color: str,
                bruises: str,
                stalk_surface_above_ring: str,
                stalk_surface_below_ring: str,
                gill_size: str,
                ring_type: str,
                population: str):

                self.odor == odor
                self.gill_color == gill_color
                self.spore_print_color == spore_print_color
                self.cap_color == cap_color
                self.bruises == bruises
                self.stalk_surface_above_ring == stalk_surface_above_ring
                self.stalk_surface_below_ring == stalk_surface_below_ring
                self.gill_size == gill_size
                self.ring_type == ring_type
                self.population == population



    def get_data_as_data_frame(self):
        try: 
            custom_data_input_dict = {
                "odor": [self.odor],
                "gill_color": [self.gill_color],
                "spore_print_color": [self.spore_print_color],
                "cap_color": [self.cap_color],
                "bruises": [self.bruises],
                "stalk_surface_above_ring": [self.stalk_surface_above_ring],
                "stalk_surface_below_ring": [self.stalk_surface_below_ring],
                "gill_size": [self.gill_size],
                "ring_type": [self.ring_type],
                "population": [self.population]
            }
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e,sys)