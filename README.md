# ❤️ Heart Disease Prediction Project

## 📌 Project Overview
This project predicts whether a patient has heart disease based on clinical features.  
We applied **data preprocessing, outlier removal, feature scaling, and multiple machine learning models** to find the most accurate classifier.

## 🗂 Dataset
- Dataset: [Heart Disease UCI Dataset](https://www.kaggle.com/ronitf/heart-disease-uci)  
- Features include: age, sex, chest pain type, cholesterol, resting BP, fasting blood sugar, max heart rate, exercise induced angina, etc.  
- Target variable: `1` = has heart disease, `0` = no heart disease.

## 🔎 Exploratory Data Analysis
- Correlation heatmap of features  
- Distribution plots of all features  
- Outlier detection & removal using the **IQR method**  

## ⚙️ Preprocessing
- Removed outliers (IQR method)  
- Standardized features using **StandardScaler**  
- Train-test split (80-20)  

## 🤖 Models Used
- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Decision Tree  
- Support Vector Classifier (SVC)  
- Random Forest (Best performing model)  

## 📊 Results
- **RandomForest Classifier** achieved **100% accuracy** on both train & test sets.  
- Cross-validation mean score: **~0.997** (very strong).  
- Confusion matrices and classification reports confirm excellent performance.

## 🚀 Prediction Example
```python
# Example new patient data
new_data = [52, 1, 0, 125, 212, 0, 1, 168, 0, 1.0, 2, 2, 3]
