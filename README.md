# Titanic Survival Prediction App 🚢

This is an end-to-end Machine Learning classification project that predicts whether a passenger survived the Titanic disaster based on passenger details.

The project includes:
- Data preprocessing
- Model training using Logistic Regression
- Model evaluation
- Deployment using Streamlit (Local + Cloud)

---

## 📌 Project Overview

The Titanic dataset is a classic binary classification problem.
The goal is to predict:

**Did the passenger survive?**
- `1` → Survived
- `0` → Did not survive

---

## 📊 Dataset Information

Dataset used: **Seaborn Titanic Dataset**

### Target Variable
- `survived`

### Input Features Used
- `sex` – Passenger gender
- `age` – Passenger age
- `pclass` – Ticket class (1, 2, 3)
- `sibsp` – Number of siblings/spouses aboard
- `parch` – Number of parents/children aboard
- `fare` – Ticket fare
- `embarked` – Port of embarkation (C, Q, S)

### Dropped Columns (to avoid data leakage)
- `alive` (duplicate of survived)
- `class`
- `who`
- `deck`
- `embark_town`
- `adult_male`
- `alone`

---

## 🧠 Machine Learning Model

- Algorithm: **Logistic Regression**
- Reason:
  - Simple
  - Interpretable
  - Suitable for binary classification

### Preprocessing Steps
- Missing value handling:
  - `age` → filled with median
  - `embarked` → filled with mode
- Categorical encoding:
  - OneHotEncoder
- Numerical scaling:
  - StandardScaler
- Pipeline used to avoid data leakage

---

## 📈 Model Performance

Accuracy achieved: **~81%**

Metrics used:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## 🗂 Project Structure

titanic-streamlit/
│
├── app.py # Streamlit application
├── model.pkl # Trained ML pipeline
├── code.ipynb # Model training & EDA
├── requirements.txt # Project dependencies
├── .gitignore
└── README.md
---

## 🚀 Running the App Locally

### Step 1: Create virtual environment
```bash
conda create -p myenv python=3.12 -y
conda activate ./myenv
Step 2: Install dependencies
pip install -r requirements.txt
Step 3: Run Streamlit app
streamlit run app.py
App will open at:
http://localhost:8501

