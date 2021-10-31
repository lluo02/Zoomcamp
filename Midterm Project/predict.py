from flask import Flask
import numpy as np
import pandas as pd
import pickle
from flask import request
from flask import jsonify
import xgboost as xgb

from sklearn.feature_extraction import DictVectorizer

# ## **Load the Model**

model_file = 'model_boost.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

# Use model:

app = Flask('churn')

@app.route('/predict', methods=['POST'])
def predict():
    employee = request.get_json()

    X = dv.transform([employee])
    dpred = xgb.DMatrix(X, feature_names=dv.get_feature_names())
    y_pred = model.predict(dpred)
    target = y_pred >= .5

    result = {
        "target_probability": float(y_pred),
        "target": bool(target)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='localhost', port=9696)