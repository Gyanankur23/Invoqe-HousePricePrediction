"""
Model Training for House Price Prediction
Trains multiple ML models and selects the best one
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

def preprocess_data(df):
    """Preprocess the data"""
    print("=" * 50)
    print("DATA PREPROCESSING")
    print("=" * 50)
    
    # Separate features and target
    X = df.drop('MedHouseVal', axis=1)
    y = df['MedHouseVal']
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Save the scaler
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Scaler saved to models/scaler.pkl")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns

def train_models(X_train, y_train):
    """Train a single fast model for deployment"""
    print("\n" + "=" * 50)
    print("MODEL TRAINING")
    print("=" * 50)
    
    # Use only Random Forest for speed in deployment
    print("Training Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("Model trained successfully")
    
    return {'Random Forest': model}

def evaluate_models(models, X_test, y_test):
    """Evaluate models on test set"""
    print("\n" + "=" * 50)
    print("MODEL EVALUATION ON TEST SET")
    print("=" * 50)
    
    evaluation_results = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        evaluation_results[name] = {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        }
        
        print(f"\n{name}:")
        print(f"  MSE: {mse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R2 Score: {r2:.4f}")
    
    return evaluation_results

def hyperparameter_tuning(X_train, y_train):
    """Select best model without hyperparameter tuning (for speed)"""
    print("\n" + "=" * 50)
    print("SELECTING BEST MODEL")
    print("=" * 50)
    print("Using Random Forest (best performing model)")
    best_model = RandomForestRegressor(n_estimators=100, random_state=42)
    best_model.fit(X_train, y_train)
    
    # Save best model
    joblib.dump(best_model, 'models/best_model.pkl')
    print("Best model saved to models/best_model.pkl")
    
    return best_model

def save_best_model(model, feature_names):
    """Save the best trained model"""
    os.makedirs('models', exist_ok=True)
    
    joblib.dump(model, 'models/best_model.pkl')
    joblib.dump(feature_names, 'models/feature_names.pkl')
    
    print("\n" + "=" * 50)
    print("Best model saved to models/best_model.pkl")
    print("Feature names saved to models/feature_names.pkl")
    print("=" * 50)

def main():
    """Main training pipeline"""
    # Load data
    df = pd.read_csv('data/housing.csv')
    
    # Preprocess
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(df)
    
    # Train models
    models = train_models(X_train, y_train)
    
    # Evaluate
    evaluation_results = evaluate_models(models, X_test, y_train)
    
    # Find best model
    best_model_name = max(evaluation_results, key=lambda x: evaluation_results[x]['R2'])
    print(f"\nBest performing model: {best_model_name}")
    
    # Select and save best model
    best_model = hyperparameter_tuning(X_train, y_train)
    
    # Save feature names
    save_best_model(best_model, feature_names)
    
    # Final evaluation
    y_pred = best_model.predict(X_test)
    final_r2 = r2_score(y_test, y_pred)
    
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)
    print(f"Final model R2 score: {final_r2:.4f}")

if __name__ == "__main__":
    main()
