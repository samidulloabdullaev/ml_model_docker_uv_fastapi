import pickle

import pandas as pd
import numpy as np

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

def data_loader(data_url:str) -> pd.DataFrame:
    
    df = pd.read_csv(data_url)

    df.columns = df.columns.str.lower().str.replace(' ', '_')

    categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

    for c in categorical_columns:
        df[c] = df[c].str.lower().str.replace(' ', '_')

    df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
    df.totalcharges = df.totalcharges.fillna(0)

    df.churn = (df.churn == 'yes').astype(int)
    
    return df

# Load the data
data_url = "https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = data_loader(data_url)

# Define features and target
y_train = df.churn

numerical = ['tenure', 'monthlycharges', 'totalcharges']
categorical = [
    'gender',
    'seniorcitizen',
    'partner',
    'dependents',
    'phoneservice',
    'multiplelines',
    'internetservice',
    'onlinesecurity',
    'onlinebackup',
    'deviceprotection',
    'techsupport',
    'streamingtv',
    'streamingmovies',
    'contract',
    'paperlessbilling',
    'paymentmethod',
]

# Convert DataFrame to list of dictionaries
train_dict = df[categorical + numerical].to_dict(orient='records')


# Create unified pipeline
pipeline = make_pipeline(
    DictVectorizer(),
    LogisticRegression(solver='liblinear')
)

# Train the model
pipeline.fit(train_dict, y_train)

# saving the model
with open('model.bin', 'wb') as f_out:
    pickle.dump(pipeline, f_out)
