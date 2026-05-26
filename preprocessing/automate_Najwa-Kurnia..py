import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

def load_data(path):
    return pd.read_csv(path, sep=';')

def handle_missing(df):
    return df.fillna(df.median(numeric_only=True))

def remove_duplicates(df):
    return df.drop_duplicates()

def normalize(df, target_col='quality'):
    scaler = MinMaxScaler()
    features = [col for col in df.columns if col != target_col]
    df[features] = scaler.fit_transform(df[features])
    return df

def save_data(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

if __name__ == '__main__':
    df = load_data('winequality_raw/winequality-red.csv')
    df = handle_missing(df)
    df = remove_duplicates(df)
    df = normalize(df)
    save_data(df, 'preprocessing/winequality_preprocessing/winequality_clean.csv')