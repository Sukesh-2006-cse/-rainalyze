# 🌧️ Rainfall Prediction using Machine Learning

## 📌 Project Overview

This project aims to build a machine learning-based system that predicts whether it will rain or not based on various atmospheric conditions. Accurate rainfall prediction is crucial for agriculture, water management, and disaster preparedness. Traditional methods can be inconsistent, and this system leverages ML algorithms to improve prediction reliability and accuracy.

---

## 🧠 Problem Statement

Accurate rainfall prediction is vital for agriculture, water resource management, and disaster preparedness. Traditional methods often lack precision due to the complexity of atmospheric data. This project aims to develop a machine learning-based model that predicts the likelihood of rainfall using weather parameters such as humidity, temperature, wind speed, and cloud cover. The goal is to improve prediction accuracy and support timely, data-driven decisions in weather forecasting.

---

## ✅ Proposed Solution

To solve this problem, we collected a historical weather dataset and preprocessed it by handling null values, removing outliers, and eliminating redundant features. Exploratory Data Analysis (EDA) was used to understand the relationships between features and rainfall patterns. The data was normalized and balanced to improve model performance and generalization.

Multiple classification models — Logistic Regression, SVC (Support Vector Classifier), and XGBoost — were trained and evaluated using metrics such as ROC-AUC and confusion matrix. The SVC model was selected based on its high validation accuracy and balanced performance.

---

## ⚙️ System Approach

### 1. System Requirements

- **OS**: Windows/Linux/MacOS  
- **Python**: 3.7 or above  
- **RAM**: 8GB minimum (16GB recommended)  
- **IDE**: Jupyter Notebook, VS Code, or Google Colab  
- **Storage**: ~1GB free disk space  
- **Internet**: For downloading dependencies and cloud integration (optional)

### 2. Libraries Required

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost imbalanced-learn
Required Libraries:

numpy

pandas

matplotlib

seaborn

scikit-learn

xgboost

imbalanced-learn

📊 Algorithm
🔎 Algorithm Selection
The Support Vector Classifier (SVC) with RBF kernel was selected for its ability to model non-linear relationships in small datasets. It was compared with Logistic Regression and XGBoost. SVC showed a strong balance between training and validation performance.

🔢 Data Input
The following features are used:

Pressure

Temperature

Dew Point

Humidity

Cloud Cover

Sunshine

Wind Direction

Wind Speed

The target variable is binary (1 = Rain, 0 = No Rain).

🏋️‍♂️ Training Process
Dataset cleaned and normalized

Redundant features removed (e.g., maxtemp, mintemp)

Balanced using RandomOverSampler

Split into 80% training, 20% validation

Trained using Logistic Regression, SVC, XGBoost

Evaluation using ROC-AUC, Confusion Matrix, Classification Report

🔮 Prediction Process
Real-time or batch input data is normalized using the trained scaler. The model then predicts the likelihood of rainfall. This can be used in dashboards, web apps, or alerts.

📈 Results
SVC Accuracy:

Training ROC-AUC: ~0.90

Validation ROC-AUC: ~0.88

Classification Report:

Precision, Recall, and F1-score all above 0.85 for positive class

Confusion Matrix:
Shows good balance in classifying both rain and no-rain scenarios.

🚀 Deployment
✅ Local Deployment
Run the notebook in Jupyter/Colab

Provide input weather values manually

Output: Rain or No Rain

🌐 Web Deployment (Optional)
Use Streamlit or Flask for UI

Host on Heroku, Render, or Streamlit Cloud

Real-time prediction using user inputs
