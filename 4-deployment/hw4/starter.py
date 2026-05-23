import argparse
import pickle
import pandas as pd

# Load the model artifacts globally
with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    """Reads and prepares the input taxi parquet data."""
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df

def run_pipeline(year: int, month: int):
    """Executes the prediction workflow for a specified year and month."""
    # 1. Build dynamic input and output names
    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'output_{year:04d}_{month:02d}.parquet'
    
    print(f"Fetching data from: {input_file}")
    
    # 2. Process data and run inference
    df = read_data(input_file)
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    # 3. Print the mean prediction value (Answer for Q5)
    mean_duration = y_pred.mean()
    print("\n" + "="*40)
    print(f"Mean predicted duration for {year:04d}-{month:02d}: {mean_duration:.2f}")
    print("="*40 + "\n")

    # 4. Generate artificial ride_id and dataframe results
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    
    df_result = pd.DataFrame({
        'ride_id': df.ride_id,
        'prediction': y_pred
    })

    # 5. Export to Parquet
    print(f"Saving results to: {output_file}")
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )

if __name__ == '__main__':
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Inference pipeline for NYC Yellow Taxi Data")
    
    parser.add_argument('--year', type=int, required=True, help="Year of taxi data (e.g., 2023)")
    parser.add_argument('--month', type=int, required=True, help="Month of taxi data (e.g., 4)")
    
    args = parser.parse_args()
    
    # Run the main pipeline with CLI arguments
    run_pipeline(year=args.year, month=args.month)

