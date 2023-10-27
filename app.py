import json
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd  
from flask_cors import CORS, cross_origin

from sklearn.preprocessing import StandardScaler
from src.pipelines.prediction_pipeline import CustomData, PredictPipeline


app=Flask(__name__)
CORS(app)

# Home Route
@app.route('/prediction', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return jsonify({"message": "getting"})

    else:
        data=CustomData(
            odor = request.form.get("odor"),
            gill_color = request.form.get("gill_color"),
            spore_print_color = request.form.get("spore_print_color"),
            cap_color = request.form.get("cap_color"),
            bruises = request.form.get("bruises"),
            stalk_surface_above_ring = request.form.get("stalk_surface_above_ring"),
            stalk_surface_below_ring = request.form.get("stalk_surface_below_ring"),
            gill_size = request.form.get("gill_size"),
            ring_type = request.form.get("ring_type"),
            population = request.form.get("population")
        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print("After Prediction")
        results=results[0]
        return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True)