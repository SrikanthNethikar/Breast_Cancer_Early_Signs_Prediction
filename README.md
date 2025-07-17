# Breast Cancer Early Signs Prediction

This is a Streamlit-based web app that predicts the risk of early signs of breast cancer using clinical and diagnostic features. The model uses a trained Random Forest classifier and includes SHAP visualizations for transparency and interpretability.

## Features
- Input form with 40 clinical and lifestyle features
- Real-time breast cancer risk prediction
- SHAP beeswarm and bar summary plots
- Individual SHAP force plot for patient-level interpretation
- Built using Python 3.10.13 and Streamlit for deployment

## Files Included
- `app.py`: Streamlit app source code
- `breast_cancer_early_signs_model.pkl`: Trained Random Forest model
- `requirements.txt`: All required dependencies
- `runtime.txt`: Python version spec (3.10.13)
- `shap_outputs/`: Contains SHAP visualizations
- `README.md`: Project documentation