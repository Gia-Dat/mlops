import os
import pickle
import click
import mlflow
import gc

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):
    mlflow.sklearn.autolog(log_models=False, log_datasets=False)
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
    
    X_train, y_train = X_train[:1000], y_train[:1000]
    X_val, y_val = X_val[:1000], y_val[:1000]

    with mlflow.start_run():
        mlflow.set_tag("developer", "ted")
        mlflow.log_param("train-data-path", "./data/green_tripdata_2023-01.parquet")
        mlflow.log_param("valid-data-path", "./data/green_tripdata_2023-02.parquet")
        mlflow.log_param("test-data-path", "./data/green_tripdata_2023-03.parquet")
        
        rf = RandomForestRegressor(max_depth=10, n_estimators=10, random_state=0)
        rf.fit(X_train, y_train)
        
        del X_train, y_train
        gc.collect()
        y_pred = rf.predict(X_val)

        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)


if __name__ == '__main__':
    run_train()
