import json
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd  
from flask_cors import CORS, cross_origin

from sklearn.preprocessing import StandardScaler
from src.pipelines.prediction_pipeline import CustomData, PredictPipeline


app=Flask(__name__)
CORS(app, origins='http://localhost:3000')

# Home Route
@app.route('/prediction', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return jsonify({"message": "getting"})

    else:
        data=CustomData(
            cap_shape = request.form.get(cap_shape),
            cap_surface = request.form.get(cap_surface),
            cap_color = request.form.get(cap_color),
            bruises = request.form.get(cap_bruises),
            odor = request.form.get(odor),
            gill_attachment = request.form.get(gill_attachment),
            gill_spacing = request.form.get(gill_spacing),
            gill_size = request.form.get(gill_size),
            gill_color = request.form.get(gill_color),
            stalk_shape = request.form.get(stalk_shape),
            stalk_root =  request.form.get(stalk_root),
            stalk_surface_above_ring = request.form.get(stalk_surface_above_ring),
            stalk_surface_below_ring = request.form.get(stalk_surface_below_ring),
            stalk_color_above_ring = request.form.get(stalk_color_above_ring),
            stalk_color_below_ring = request.form.get(stalk_color_below_ring),
            veil_type =  request.form.get(veil_type),
            veil_color = request.form.get(veil_color),
            ring_number = request.form.get(ring_number),
            ring_type = request.form.get(ring_type),
            spore_print_color = request.form.get(spore_print_color),
            population = request.form.get(population),
            habitat = request.form.get(habitat)
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