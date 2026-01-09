# Red Wine Quality Prediction and Deployment

This project automates wine quality prediction using machine learning, providing accurate and efficient insights for quality assurance.

## Key Features
- **Exploratory Data Analysis (EDA)**: Comprehensive data cleaning including outlier removal and duplicate handling.
- **Feature Engineering**: Correlation analysis and selection of high-impact features.
- **SMOTE Balancing**: Addressed class imbalance to ensure robust training.
- **XGBoost Classifier**: Selected as the best performing model after 10-fold cross-validation.
- **Premium UI**: Modern Flask application with glassmorphism design.

## Technical Stack
- **Languages**: Python, HTML, CSS
- **ML Libraries**: Scikit-Learn, XGBoost, Pandas, NumPy, Imbalanced-Learn
- **Web Framework**: Flask
- **Deployment**: Render / Gunicorn

## Performance Metrics
- **Accuracy**: ~93.6% (Test Set)
- **F1-Score**: ~0.94
- **Precision (Good Wine)**: ~0.92
- **Recall (Good Wine)**: ~0.95

## How to Run Locally
1. Install dependencies: `pip install -r requirements.txt`
2. Run training script: `python train.py`
3. Start Flask app: `python app.py`
4. Visit `http://127.0.0.1:5000`
