"""
Download House Price Dataset from Kaggle
Dataset: California Housing Dataset (from sklearn - real public dataset)
This is a real-world dataset from the 1990 California census
"""

import pandas as pd
from sklearn.datasets import fetch_california_housing
import os

def download_dataset():
    """Download and save California Housing dataset"""
    print("Downloading California Housing dataset (real public dataset)...")
    
    # Fetch dataset from sklearn (real California census data)
    housing = fetch_california_housing()
    
    # Create DataFrame
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['MedHouseVal'] = housing.target
    
    # Save to CSV
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/housing.csv', index=False)
    
    print(f"Dataset saved to data/housing.csv")
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nFirst few rows:\n{df.head()}")
    
    print("\nDataset Info:")
    print("- Source: 1990 California Census (Real public dataset)")
    print("- Samples: 20,640 districts")
    print("- Features: 8 numerical features")
    print("- Target: Median house value (in $100,000s)")
    
    return df

if __name__ == "__main__":
    download_dataset()
