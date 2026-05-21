import os
import pickle

import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

import mlflow

url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet'

def read_dataframe(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df

def create_X(df, dv=None):
    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient="records")

    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X= dv.transform(dicts)

    return X, dv

def train_model(X_train, y_train, dv):
    
    mlflow.set_experiment("nyc-taxi-experiment")
    
    with mlflow.start_run() as run:
        
        lr = LinearRegression()
        mlflow.log_param("model_type", "LinearRegression")
        lr.fit(X_train, y_train)
        
        print(f"Intercept: {lr.intercept_:.2f}")

        os.makedirs("models", exist_ok=True)
        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        mlflow.sklearn.log_model(lr, artifact_path="models_mlflow")
        
        return dv, lr, run.info.run_id

if __name__ == '__main__':
    df = read_dataframe(url)    
    X_train, dv = create_X(df)
    y_train = df['duration'].values
    dv, lr, run_id = train_model(X_train, y_train, dv)

