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
                cap_shape: str,
                cap_surface: str,
                cap_color: str,
                bruises: str,
                odor: str,
                gill_attachment: str,
                gill_spacing: str,
                gill_size: str,
                gill_color: str,
                stalk_shape: str,
                stalk_root: str,
                stalk_surface_above_ring: str,
                stalk_surface_below_ring: str,
                stalk_color_above_ring: str,
                stalk_color_below_ring: str,
                veil_type: str,
                veil_color: str,
                ring_number: str,
                ring_type: str,
                spore_print_color: str,
                population: str,
                habitat: str):

                self.cap-shape == cap_shape
                self.cap-surface == cap_surface
                self.cap-color == cap_color
                self.bruises == cap_bruises
                self.odor == odor
                self.gill-attachment == gill_attachment
                self.gill-spacing == gill_spacing
                self.gill-size == gill_size
                self.gill-color == gill_color
                self.stalk-shape == stalk_shape
                self.stalk-root ==  stalk_root
                self.stalk-surface-above-ring == stalk_surface_above_ring
                self.stalk-surface-below-ring == stalk_surface_below_ring
                self.stalk-color-above-ring == stalk_color_above_ring
                self.stalk-color-below-ring == stalk_color_below_ring
                self.veil-type ==  veil_type
                self.veil-color == veil_color
                self.ring-number == ring_number
                self.ring-type == ring_type
                self.spore-print-color == spore_print_color
                self.population == population
                self.habitat == habitat

    def get_data_as_data_frame(self):
        try: 
            custom_data_input_dict = {
                "cap_shape": [self.cap-shape], 
                "cap_surface": [self.cap-surface],
                "cap_color": [self.cap-color],
                "cap_bruises": [self.bruises],
                "odor": [self.odor],
                "gill_attachment": [self.gill-attachment],
                "gill_spacing": [self.gill-spacing], 
                "gill_size": [self.gill-size], 
                "gill_color": [self.gill-color], 
                "stalk_shape": [self.stalk-shape], 
                "stalk_root": [self.stalk-root], 
                "stalk_surface_above_ring": [self.stalk-surface-above-ring],
                "stalk_surface_below_ring": [self.stalk-surface-below-ring], 
                "stalk_color_above_ring": [self.stalk-color-above-ring], 
                "stalk_color_below_ring": [self.stalk-color-below-ring],
                "veil_type": [self.veil-type],  
                "veil_color": [self.veil-color], 
                "ring_number": [self.ring-number], 
                "ring_type": [self.ring-type] ,
                "spore_print_color": [self.spore-print-color], 
                "population": [self.population], 
                "habitat": [self.habitat],
            }
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e,sys)