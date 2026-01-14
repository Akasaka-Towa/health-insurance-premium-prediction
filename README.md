# Health Insurance Premium Prediction using Machine Learning

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Educational-green?style=for-the-badge)

## 📌 Project Overview
This project predicts **monthly health insurance premiums (INR)** using machine learning models based on an individual's **demographic, lifestyle, and medical attributes**.

The project demonstrates a complete **end-to-end machine learning workflow**, including dataset generation, preprocessing, model training, evaluation, and deployment using a **Streamlit web application**.

---

## 🎯 Problem Statement
Health insurance premiums depend on multiple risk factors such as age, BMI, smoking habits, medical history, and lifestyle choices.
The objective of this project is to build a reliable system that can **estimate insurance premiums** using regression-based machine learning techniques.

---

## 🧠 Machine Learning Models Used
The following models are implemented and compared in this project:

* **Linear Regression**
* **Lasso Regression** (L1 Regularization)
* **Ridge Regression** (L2 Regularization)
* **Support Vector Regression** (SVR – RBF Kernel)
* **PCA + Linear Regression**

All models are implemented using **scikit-learn pipelines** with standard feature scaling.

---

## 📊 Dataset Description
The dataset is **synthetically generated** to simulate real-world health insurance data.

### 🔹 Input Features (20)
| Category | Features |
| :--- | :--- |
| **Demographics** | Age, Gender, Number of Children, Annual Income |
| **Lifestyle** | BMI, Smoking Status, Exercise Hours/Week, Diet Quality Score, Alcohol Consumption, Sleep Duration |
| **Medical History** | Chronic Conditions, Hospital Visits, Monthly Medication Cost, Blood Pressure, Cholesterol Level |
| **Preventive Care** | Preventive Health Checkups, Family Medical History, Dental Visits, Preventive Screenings |
| **Psychological** | Stress Level |

### 🔹 Target Variable
* `monthly_premium_inr`

---

## 📂 Project Structure
```text
health_insurance_project/
│
├── data/
│   └── synthetic_health_insurance_20_features.csv  # Generated Dataset
│
├── src/
│   ├── generate_dataset.py       # Script to create synthetic data
│   └── train_health.py           # ML Pipeline & Model Training
│
├── figures/
│   ├── pred_vs_actual_linear.png # Visualization results
│   ├── pred_vs_actual_svr.png
│   └── pca_scree.png
│
├── outputs/
│   ├── metrics_summary.json      # Model performance metrics
│   ├── lasso_coefficients.csv
│   ├── ridge_coefficients.csv
│   └── user_predictions.csv      # Saved user inputs from App
│
├── app.py                        # Streamlit Application
├── requirements.txt              # Project Dependencies
└── README.md                     # Project Documentation

```

---

## ▶️ How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone [https://github.com/Akasaka-Towa/health-insurance-prediction.git](https://github.com/Akasaka-Towa/health-insurance-prediction.git)
cd health-insurance-prediction

```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt

```

### 3️⃣ Generate dataset

*(Optional: Only if data is missing)*

```bash
python src/generate_dataset.py

```

### 4️⃣ Train all models

```bash
python src/train_health.py

```

### 5️⃣ Run the Streamlit web app

```bash
python -m streamlit run app.py

```

The application will open in your browser at: `http://localhost:8501`

---

## 🌐 Streamlit Web Application

The Streamlit app provides an interactive interface to:

* Select a machine learning model.
* Enter personal and health details via a user-friendly form.
* Predict monthly insurance premium in real-time.
* View prediction ranges and detailed model information.
* Save predictions to a local CSV file.

> **Note:** Add a screenshot of your application here to show how it looks!
> `![App Screenshot](figures/app_screenshot.png)`

---

## 📈 Results Summary

* **Linear, Ridge, and Lasso regression** models achieved the best overall performance.
* **SVR** captured non-linear patterns but showed higher error rates on this dataset.
* **PCA** successfully reduced dimensionality with a small trade-off in accuracy.

---

## 🛠 Technologies Used

* **Python 3.12**
* **Data Science:** NumPy, Pandas, Matplotlib
* **Machine Learning:** Scikit-learn
* **Web Framework:** Streamlit
* **Version Control:** Git & GitHub

---

## 👥 Authors & Contributors

**Lead Author:**

* **[Akasaka-Towa](https://www.google.com/search?q=https://github.com/Akasaka-Towa)**

**Contributors:**

* Aryan Sinha
* Atharv Gupta
* Atul Bhat
* Charchit Jain
* Vivek Kumar

---

## 📜 License

This project is intended for **educational and academic purposes only**.

```
