# Enhanced Medical Delivery Mode Prediction System

A machine learning-based solution for predicting childbirth delivery modes (cesarean or vaginal) using maternal and fetal data, achieving 92-97% accuracy. This project features an XGBoost model, synthetic data generation, and an interactive Streamlit web application for real-time risk assessment.

## Features
- **High Accuracy**: Predicts delivery modes with \( 92\% - 97\% \) accuracy using XGBoost.
- **Data Preprocessing**: Handles missing values, scales features, and balances classes with SMOTE.
- **Synthetic Data**: Generates realistic medical datasets for training and testing.
- **Validation**: Uses 5-fold cross-validation for robust performance evaluation.
- **Interactive Interface**: Streamlit app with prediction, data analysis, and visualization tools.
- **Risk Visualization**: Includes gauge charts and statistical plots for intuitive insights.

---

## Installation

### Prerequisites
- Python 3.8+
- Git

### Steps
1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/enhanced-medical-delivery-prediction.git
   cd enhanced-medical-delivery-prediction
   ```
---
## Usage

### Training the Model
Run the main script to train the model and save it:
```bash
python medical_predictor.py
```
- Generates 10,000 synthetic samples.
- Trains an XGBoost model.
- Saves the model to `medical_models/`.
- Displays performance metrics (accuracy, ROC-AUC, etc.).

### Running the Web App
Launch the Streamlit application:
```bash
streamlit run app.py
```
Visit `http://localhost:8501` in your browser to use the app.

### Making Predictions
Predict delivery modes for new data programmatically:
```python
import pandas as pd
import joblib

# Load saved model
model_data = joblib.load('medical_models/model_v1.0.pkl')
predictor = model_data['model']
preprocessor = model_data['preprocessor']

# New patient data
new_data = pd.DataFrame({...})  # See code for structure
predictions = predictor.predict(preprocessor.transform(new_data))
```

---

## Model Details
- **Algorithm**: XGBoost Classifier
- **Hyperparameters**:
  - `n_estimators=200`
  - `max_depth=6`
  - `learning_rate=0.01`
  - `subsample=0.8`
  - `colsample_bytree=0.8`
  - `min_child_weight=3`
- **Preprocessing**:
  - Numeric: Median imputation, StandardScaler
  - Categorical: One-hot encoding
  - Class balancing: SMOTE
- **Features**: Maternal age, BMI, blood pressure, fetal heart rate, etc.
- **Evaluation**: 5-fold cross-validation, accuracy (\( 92\% - 97\% \)), ROC-AUC, classification report

---

## Web Application
The Streamlit app includes three tabs:
1. **Prediction**: Enter patient data for delivery mode predictions with probability gauges.
2. **Data Analysis**: Explore synthetic data statistics and visualizations (histograms, box plots).
3. **About**: Project overview and usage notes.

---

