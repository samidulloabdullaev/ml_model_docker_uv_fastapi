import pickle

import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


def data_loader(data_url: str) -> pd.DataFrame:
    df = pd.read_csv(data_url)

    df.columns = df.columns.str.lower().str.replace(" ", "_")

    categorical_columns = list(df.dtypes[df.dtypes == "object"].index)

    for c in categorical_columns:
        df[c] = df[c].str.lower().str.replace(" ", "_")

    df.totalcharges = pd.to_numeric(df.totalcharges, errors="coerce")
    df.totalcharges = df.totalcharges.fillna(0)

    df.churn = (df.churn == "yes").astype(int)

    return df


def train_model(df: pd.DataFrame) -> None:
    # Define features and target
    y_train = df.churn

    numerical = ["tenure", "monthlycharges", "totalcharges"]
    categorical = [
        "gender",
        "seniorcitizen",
        "partner",
        "dependents",
        "phoneservice",
        "multiplelines",
        "internetservice",
        "onlinesecurity",
        "onlinebackup",
        "deviceprotection",
        "techsupport",
        "streamingtv",
        "streamingmovies",
        "contract",
        "paperlessbilling",
        "paymentmethod",
    ]

    # Convert DataFrame to list of dictionaries
    train_dict = df[categorical + numerical].to_dict(orient="records")

    # Create unified pipeline
    pipeline = make_pipeline(DictVectorizer(), LogisticRegression(solver="liblinear"))

    # Train the model
    pipeline.fit(train_dict, y_train)
    return pipeline


def save_model(model, filename: str = "model.bin") -> None:
    """Save the model to a file using pickle"""
    with open(filename, "wb") as f_out:
        pickle.dump(model, f_out)

    print(f"The model is saved to {filename}")


# Load the data
data_url = "https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = data_loader(data_url)

pipeline = train_model(df)

save_model(pipeline, "model.bin")
