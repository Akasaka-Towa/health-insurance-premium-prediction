# Health Insurance Premium Prediction using Machine Learning

## 📌 Project Overview
This project predicts **monthly health insurance premiums (INR)** using multiple machine learning models based on an individual's demographic, lifestyle, and medical attributes.

The system demonstrates the complete **ML pipeline**:
- Synthetic data generation
- Data preprocessing
- Model training & evaluation
- Visualization
- Deployment-ready web application (Streamlit)

---

## 🎯 Problem Statement
Health insurance premiums depend on multiple risk factors such as age, BMI, smoking habits, medical history, and lifestyle.  
This project aims to **predict insurance premiums accurately** using regression-based machine learning models.

---

## 🧠 Machine Learning Models Used
- Linear Regression
- Lasso Regression (L1 Regularization)
- Ridge Regression (L2 Regularization)
- Support Vector Regression (RBF Kernel)
- PCA + Linear Regression

All models are implemented using **scikit-learn pipelines** with feature scaling.

---

## 📊 Dataset Description
The dataset is **synthetically generated** and contains **1000 samples** with the following:

### Input Features (20)
- Age, BMI, Gender
- Smoking status
- Income
- Exercise & diet scores
- Stress level
- Blood pressure & cholesterol
- Medical history indicators
- Preventive health checkups

### Target Variable
- `monthly_premium_inr`

---

## 📂 Project Structure
