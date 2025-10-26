# 🥚 Poultry Egg Production Predictor

An AI-powered Streamlit web app that predicts daily egg production in poultry farms based on feeding and environmental factors.

## 🚀 Features
- Real-time prediction using Random Forest Regression
- Automatic confidence feedback for farm conditions
- Interactive feature importance visualization
- Clean, simple Streamlit UI

## 🧠 Model Summary
- **Algorithm:** Random Forest Regressor
- **R²:** 0.974
- **MAE:** 19.71
- **RMSE:** 33.86

## 🧩 How It Works
The model uses both direct farm inputs and derived features:
- **Feed per Chicken = Feeding / Number of Chickens**
- **Environmental Stress Index = Temperature × Humidity × Ammonia**


## ⚙️ How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
