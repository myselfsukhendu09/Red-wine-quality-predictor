# Red Wine Quality Prediction (Jupyter Notebook)

This project automates wine quality prediction using machine learning, providing accurate and efficient insights for quality assurance. This repository contains the complete analysis and model development within a Jupyter Notebook.

## 🚀 Overview
The project uses physicochemical properties of red wine to predict its quality. It includes comprehensive data cleaning, feature engineering with SMOTE balancing, and a tuned XGBoost classifier.

## 📓 Features
- **End-to-End Notebook**: Data collection, EDA, and model training in a single file.
- **Interactive Dashboard**: Built-in sliders and input fields (using `ipywidgets`) for real-time quality prediction.
- **Advanced ML**: Utilizes XGBoost with hyperparameter tuning via GridSearchCV.
- **Class Balancing**: Handles minority classes using SMOTE.

## 🛠️ How to Use
### Option 1: Google Colab (Recommended)
1. Open [Google Colab](https://colab.research.google.com).
2. Go to the **GitHub** tab.
3. Paste the repository URL: `https://github.com/myselfsukhendu09/Red-wine-quality-predictor`.
4. Click on `Red_Wine_Quality_Prediction.ipynb`.

### Option 2: Local Deployment
1. Clone the repository:
   ```bash
   git clone https://github.com/myselfsukhendu09/Red-wine-quality-predictor.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch Streamlit Application:
   ```bash
   streamlit run streamlit_app.py
   ```
4. Or use Jupyter: `jupyter notebook Red_Wine_Quality_Prediction.ipynb`

## 📊 Performance
- **Test Accuracy**: ~94%
- **Model**: XGBoost (optimized)
- **Features Analyzed**: Volatile Acidity, Residual Sugar, Chlorides, Total Sulfur Dioxide, Density, Sulphates, Alcohol.
