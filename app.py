import joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load model and scaler
model = joblib.load("random_forest_model.pkl")

# Page configuration
st.set_page_config(
    page_title="Poultry Egg Production Predictor",
    page_icon="🥚",
    layout="wide"
)

st.title("🐔 Poultry Egg Production Prediction Dashboard")
st.write("""
Use this AI-powered tool to predict daily egg production in poultry farms based on feeding and environmental data.
""")

# Tabs for organization
tab1, tab2, tab3 = st.tabs(["🔢 Prediction", "📊 Model Performance", "🌿 Feature Importance"])

# ------------------------------------------------------------------
# 🔢 TAB 1: Prediction
# ------------------------------------------------------------------
with tab1:
    st.header("Enter Farm Data")
    col1, col2 = st.columns(2)

    with col1:
        amount_of_chicken = st.number_input("Amount of Chickens", min_value=100, max_value=5000, value=2700)
        amount_of_feeding = st.number_input("Feed Quantity (kg)", min_value=50.0, max_value=500.0, value=180.0)
        ammonia = st.number_input("Ammonia Level (ppm)", min_value=5.0, max_value=25.0, value=15.0)
        temperature = st.number_input("Temperature (°C)", min_value=20.0, max_value=40.0, value=30.0)

    with col2:
        humidity = st.number_input("Humidity (%)", min_value=20.0, max_value=100.0, value=50.0)
        light_intensity = st.number_input("Light Intensity (lux)", min_value=100.0, max_value=500.0, value=330.0)
        noise = st.number_input("Noise Level (dB)", min_value=50.0, max_value=300.0, value=200.0)

    # Derived features
    feed_per_chicken = round(amount_of_feeding / amount_of_chicken, 5)
    env_stress_index = round((temperature * humidity * ammonia), 3)

    st.subheader("🧩 Derived Features")
    st.write(f"**Feed per Chicken:** {feed_per_chicken}")
    st.write(f"**Environmental Stress Index:** {env_stress_index}")

    # Prepare input data
    features = pd.DataFrame({
        'amount_of_chicken': [amount_of_chicken],
        'amount_of_feeding': [amount_of_feeding],
        'ammonia': [ammonia],
        'temperature': [temperature],
        'humidity': [humidity],
        'light_intensity': [light_intensity],
        'noise': [noise],
        'feed_per_chicken': [feed_per_chicken],
        'environmental_stress_index': [env_stress_index]
    })

    # Model prediction
    if st.button("Predict Egg Production 🥚"):
        prediction = model.predict(features)[0]
        st.success(f"Estimated Egg Production: **{prediction:.0f} eggs/day**")
        st.balloons()

        # Confidence and validation
        if not (28 <= temperature <= 32 and 170 <= amount_of_feeding <= 200):
            st.warning(
                "⚠️ Note: These input values fall outside the model's optimal training range "
                "(Temperature 28–32°C, Feeding 170–200 kg). Prediction confidence may be lower."
            )
        else:
            st.info("✅ Inputs within optimal range. Prediction confidence: **High**")

# ------------------------------------------------------------------
# 📊 TAB 2: Model Performance Info
# ------------------------------------------------------------------
with tab2:
    st.header("Model Evaluation Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", "0.974")
    col2.metric("MAE", "19.71")
    col3.metric("RMSE", "33.86")

    st.markdown("""
    **Interpretation:**
    - The model explains **97.4% of the variance** in egg production.
    - The **average error** is around **20 eggs per prediction**, suitable for real-world use.
    """)

# ------------------------------------------------------------------
# 🌿 TAB 3: Feature Importance
# ------------------------------------------------------------------
with tab3:
    st.header("Feature Importance Analysis")
    importances = model.feature_importances_
    feature_names = [
        'amount_of_chicken', 'amount_of_feeding', 'ammonia',
        'temperature', 'humidity', 'light_intensity', 'noise',
        'feed_per_chicken', 'environmental_stress_index'
    ]
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by="Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(importance_df['Feature'], importance_df['Importance'])
    ax.set_xlabel("Feature Importance")
    ax.set_title("Random Forest Feature Contributions")
    st.pyplot(fig)

# Footer
st.markdown("---")
st.caption("Developed with ❤️ using Random Forest Regression and Streamlit")
