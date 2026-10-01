"""
Streamlit App for House Price Prediction
Interactive interface for predicting house prices
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set page config
st.set_page_config(page_title="House Price Prediction", page_icon="🏠")

# Title
st.title("🏠 California House Price Prediction")
st.markdown("Predict median house values in California districts using Machine Learning")

# Load model and scaler
def load_model():
    """Load the trained model and scaler, or train if not found"""
    try:
        model = joblib.load('models/best_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        return model, scaler, feature_names
    except:
        st.warning("Model not found. Training model now... This may take a moment.")
        try:
            # Download data
            from data.download_data import download_california_housing
            download_california_housing()
            
            # Train model
            from train_model import main as train_main
            train_main()
            
            # Load the newly trained model
            model = joblib.load('models/best_model.pkl')
            scaler = joblib.load('models/scaler.pkl')
            feature_names = joblib.load('models/feature_names.pkl')
            return model, scaler, feature_names
        except Exception as e:
            st.error(f"Error training model: {str(e)}")
            return None, None, None

model, scaler, feature_names = load_model()

if model is None:
    st.stop()

# Input form
st.header("Enter House Features")

col1, col2 = st.columns(2)

with col1:
    MedInc = st.number_input("Median Income (in $10,000s)", min_value=0.0, max_value=15.0, value=3.0, step=0.1)
    HouseAge = st.number_input("House Age (years)", min_value=0, max_value=100, value=20)
    AveRooms = st.number_input("Average Rooms", min_value=1.0, max_value=20.0, value=5.0, step=0.1)
    AveBedrms = st.number_input("Average Bedrooms", min_value=0.5, max_value=10.0, value=1.0, step=0.1)

with col2:
    Population = st.number_input("Population", min_value=100, max_value=50000, value=1000, step=100)
    AveOccup = st.number_input("Average Occupancy", min_value=1.0, max_value=10.0, value=3.0, step=0.1)
    Latitude = st.number_input("Latitude", min_value=32.0, max_value=42.0, value=37.0, step=0.1)
    Longitude = st.number_input("Longitude", min_value=-125.0, max_value=-114.0, value=-120.0, step=0.1)

# Predict button
if st.button("Predict Price"):
    # Create feature array
    features = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]])
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict
    prediction = model.predict(features_scaled)[0]
    
    # Display result
    st.success(f"Predicted Median House Value: ${prediction * 100000:,.2f}")
    
    # Display feature importance
    if hasattr(model, 'feature_importances_'):
        st.subheader("Feature Importance")
        importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        st.bar_chart(importance.set_index('Feature'))

st.markdown("---")
st.markdown("### Model Information")
st.info("Model: Random Forest Regressor\nDataset: California Housing Dataset\nAccuracy: R² ≈ 0.80")
