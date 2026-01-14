# Health Insurance Premium Prediction using Machine Learning

## 📌 Project Overview
This project predicts **monthly health insurance premiums (INR)** using machine learning models based on an individual's **demographic, lifestyle, and medical attributes**.

The project demonstrates a complete **end-to-end machine learning workflow**, including dataset generation, preprocessing, model training, evaluation, and deployment using a **Streamlit web application**.

---

## 🎯 Problem Statement
Health insurance premiums depend on multiple risk factors such as age, BMI, smoking habits, medical history, and lifestyle choices.  
The objective of this project is to build a reliable system that can **estimate insurance premiums** using regression-based machine learning techniques.

---

## 🧠 Machine Learning Models Used
The following models are implemented and compared:

- Linear Regression  
- Lasso Regression (L1 Regularization)  
- Ridge Regression (L2 Regularization)  
- Support Vector Regression (SVR – RBF Kernel)  
- PCA + Linear Regression  

All models are implemented using **scikit-learn pipelines** with feature scaling.

---

## 📊 Dataset Description
The dataset is **synthetically generated** to simulate real-world health insurance data.

### 🔹 Input Features (20)
- Age  
- BMI  
- Gender  
- Smoking status  
- Number of children  
- Annual income  
- Exercise hours per week  
- Diet quality score  
- Stress level  
- Chronic conditions  
- Hospital visits  
- Monthly medication cost  
- Alcohol consumption  
- Sleep duration  
- Blood pressure  
- Cholesterol level  
- Preventive health checkups  
- Family medical history  
- Dental visits  
- Preventive screenings  

### 🔹 Target Variable
- `monthly_premium_inr`

---

## 📂 Project Structure
```text
health_insurance_project/
│
├── data/
│   └── synthetic_health_insurance_20_features.csv
│
├── src/
│   ├── generate_dataset.py
│   └── train_health.py
│
├── figures/
│   ├── pred_vs_actual_linear.png
│   ├── pred_vs_actual_svr.png
│   └── pca_scree.png
│
├── outputs/
│   ├── metrics_summary.json
│   ├── lasso_coefficients.csv
│   ├── ridge_coefficients.csv
│   └── user_predictions.csv
│
├── app.py
├── requirements.txt
└── README.md

---

## ▶️ How to Run the Project

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt

###2️⃣ Generate dataset (if not already present)
python src/generate_dataset.py

###3️⃣ Train all models
python src/train_health.py

###4️⃣ Run the Streamlit web app
python -m streamlit run app.py


### The application will open in your browser at:

http://localhost:8501

## 🌐 Streamlit Web Application

The Streamlit app allows users to:

- Select a machine learning model  
- Enter personal and health details  
- Predict monthly insurance premium in real time  
- View prediction range and model information  
- Save predictions to a CSV file  

---

## 📈 Results Summary

- Linear, Ridge, and Lasso regression models achieved the best overall performance  
- SVR captured non-linear patterns but showed higher error  
- PCA reduced dimensionality with a small trade-off in accuracy  

---

## 🛠 Technologies Used

- Python 3.12  
- NumPy  
- Pandas  
- Matplotlib  
- Scikit-learn  
- Streamlit  
- Git & GitHub  

---

## 👤 Author

**Akasaka-Towa**

### Contributors
- Aryan Sinha
- Atharv Gupta  
- Atul Bhat  
- Charchit Jain  
- Vivek Kumar  


---

## 📜 License

This project is intended for **educational and academic purposes only**.

