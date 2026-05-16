#!/usr/bin/env python
from datetime import datetime
import pickle
from pathlib import Path
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
import mlflow

# Import Airflow decorators
from airflow.decorators import dag, task

# --- Keep your core processing logic intact ---
def read_dataframe(year, month):
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(url)
    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    return df

def create_X(df, dv=None):
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient="records")
    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)
    return X, dv

# --- Define the Airflow DAG ---
@dag(
    dag_id="nyc_taxi_mlflow_pipeline",
    schedule="@monthly",               # Step 4: Schedule monthly
    start_date=datetime(2025, 1, 1),   # Allows us to backfill historical data
    catchup=False,                     # Prevent it from running instantly for every month since 2025
    tags=["mlops", "xgboost"]
)
def taxi_pipeline():

    @task
    def run_training_pipeline(**context):
        # Step 4: Airflow provides the exact execution context date automatically
        logical_date = context['logical_date']
        
        # Calculate training (2 months ago) and validation (1 month ago) relative to execution time
        train_date = logical_date.subtract(months=2)
        val_date = logical_date.subtract(months=1)
        
        print(f"Training using data from: {train_date.year}-{train_date.month}")
        print(f"Validating using data from: {val_date.year}-{val_date.month}")
        
        # 1. Load Data
        df_train = read_dataframe(year=train_date.year, month=train_date.month).sample(1000)
        df_val = read_dataframe(year=val_date.year, month=val_date.month).sample(1000)

        # 2. Vectorize Features
        X_train, dv = create_X(df_train)
        X_val, _ = create_X(df_val, dv)

        y_train = df_train['duration'].values
        y_val = df_val['duration'].values

        # 3. Train & Log with MLflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("nyc-taxi-experiment")
        
        Path('models').mkdir(exist_ok=True)

        with mlflow.start_run() as run:
            train = xgb.DMatrix(X_train, label=y_train)
            valid = xgb.DMatrix(X_val, label=y_val)

            best_params = {
                'learning_rate': 0.09585355369315604,
                'max_depth': 30,
                'min_child_weight': 1.060597050922164,
                'objective': 'reg:linear',
                'reg_alpha': 0.018060244040060163,
                'reg_lambda': 0.011658731377413597,
                'seed': 42
            }

            mlflow.log_params(best_params)

            booster = xgb.train(
                params=best_params,
                dtrain=train,
                num_boost_round=30,
                evals=[(valid, 'validation')],
                early_stopping_rounds=50
            )

            y_pred = booster.predict(valid)
            rmse = root_mean_squared_error(y_val, y_pred)
            mlflow.log_metric("rmse", rmse)

            # Save & track artifacts
            with open("models/preprocessor.b", "wb") as f_out:
                pickle.dump(dv, f_out)
            mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")
            mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")
            
            print(f"Successfully logged MLflow run: {run.info.run_id}")
            
            with open("run_id.txt", "w") as f:
                f.write(run.info.run_id)

    # Execute the task inside the DAG
    run_training_pipeline()

# Instantiate the DAG
dag_instance = taxi_pipeline()