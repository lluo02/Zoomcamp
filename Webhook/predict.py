from flask import Flask
import numpy as np
import pandas as pd
import pickle
from flask import request
from flask import jsonify

from sklearn.feature_extraction import DictVectorizer

# ## **Load the Model**

model_file = 'model_C=1.0.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

# Use model:

app = Flask('churn')

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= .5

    result = {
        "churn_probability": float(y_pred),
        "churn": bool(churn)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='localhost', port=9696)