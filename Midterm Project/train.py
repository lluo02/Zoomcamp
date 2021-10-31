import numpy as np
import pandas as pd
import pickle
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.utils import class_weight
from sklearn.metrics import roc_auc_score

import xgboost as xgb

# Parameters

xgb_params = {
    'eta': 0.1,
    'max_depth': 6,
    'min_child_weight': 25, 
    
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'nthread': 8,
    
    'seed': 1,
    'verbosity': 1,
}

weights = {0:.25, 1:.75}

# Data prep

df = pd.read_csv('aug_train.csv')

categorical_columns = [
    'city',
    'gender',
    'relevent_experience',
    'enrolled_university',
    'education_level',
    'major_discipline',
    'experience',
    'company_size',
    'company_type',
    'last_new_job',
]

numerical = [
    'city_development_index',
    'training_hours'
]

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')
    
# Remove XGB-incompatible strings
df['company_size'] = df['company_size'].str.replace('<10', '0')
df['experience'] = df['experience'].str.replace('<1', '0')

# Unnecessary to training
del df['enrollee_id']

# Null values will be assigned to the mode.
has_null = [
    'gender', 'enrolled_university', 'education_level', 'major_discipline',
    'experience', 'company_size', 'company_type', 'last_new_job'
]
for c in has_null:
    df[c].fillna(df[c].mode().value_counts().index[0], inplace=True)
    
# Splitting and one-hot encoding
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
y_full_train = df_full_train.target.values
df_full_train = df_full_train.drop('target', axis=1)
y_test = df_test.target.values
df_test = df_test.drop('target', axis=1)

dicts_full_train = df_full_train[categorical_columns + numerical].to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(dicts_full_train)
dicts_test = df_test[categorical_columns + numerical].to_dict(orient='records')
X_test = dv.transform(dicts_test)

# Calculating sample weight objects to account for dataset imbalance
classes_weights_full_train = class_weight.compute_sample_weight(
    class_weight=weights,
    y=y_full_train
)
classes_weights_test = class_weight.compute_sample_weight(
    class_weight=weights,
    y=y_test
)

# Train the model
dfulltrain = xgb.DMatrix(X_full_train, label=y_full_train, feature_names=dv.get_feature_names(), weight=classes_weights_full_train)
dtest = xgb.DMatrix(X_test, feature_names=dv.get_feature_names())

boost = xgb.train(xgb_params, dfulltrain, num_boost_round=30)

# Test the model
y_pred = boost.predict(dtest)
print("aoc is", roc_auc_score(y_test, y_pred, sample_weight=classes_weights_test))

# ## **Save the model**

output_file = f'model_boost.bin'

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, boost), f_out)
    # Do stuff

# Do other stuff
print(f'model saved to {output_file}')