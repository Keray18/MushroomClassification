import os 
import sys   
from dataclasses import dataclass

from sklearn.ensemble import (GradientBoostingClassifier, RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, eval_model


@dataclass
class ModelTrainingConfig:
    trained_model_file_path = os.path.join('artifacts', "model.pkl")


class ModelTraining:
    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()

    def initiate_model_training(self, train_arr, test_arr):
        try:
            logging.info("Splitting training and testing data")
            X_train, y_train, X_test, y_test=(train_arr[:,:-1], train_arr[:,-1], test_arr[:,:-1], test_arr[:,-1])

            models = {
                "Gradient Boosting": GradientBoostingClassifier(), 
                "Random Forest": RandomForestClassifier(),
                "Logistic Regression": LogisticRegression(),
                "Decision Tree": DecisionTreeClassifier(),
                "XGBoost": XGBClassifier(),
            }

            logging.info("Getting the model report...")
            model_report:dict=eval_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No Best Model Found")
            logging.info(f"The Best Model found is: {best_model_name}")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            accuracy=accuracy_score(y_test, predicted)
            return accuracy

        except Exception as e:
            raise CustomException(e,sys)