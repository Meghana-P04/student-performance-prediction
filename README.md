# 🎓 Student Performance Prediction

A Machine Learning project that predicts a student's academic performance level — **Low, Average, or High** — based on academic, personal, and educational factors.

## 📌 Project Overview

Student academic performance can be influenced by several factors such as study hours, attendance, previous scores, parental involvement, access to resources, motivation, tutoring, and other educational and personal characteristics.

This project analyzes these factors and builds a machine learning pipeline to predict the student's overall performance level.

The project also includes a **Streamlit web application** where users can enter student information and receive a performance prediction.

---

## 🎯 Objectives

- Analyze factors affecting student academic performance.
- Perform Exploratory Data Analysis (EDA).
- Clean and preprocess the dataset.
- Handle categorical and numerical features.
- Create performance categories from exam scores.
- Train and compare multiple machine learning models.
- Select a suitable performing model.
- Save the trained model and preprocessing objects.
- Build an interactive Streamlit application for predictions.

---

## 📊 Dataset

The dataset contains:

- **Number of rows:** 6,607
- **Number of columns:** 20
- **Target variable:** `Exam_Score`

### Features

| Feature | Description |
|---|---|
| Hours_Studied | Number of hours spent studying |
| Attendance | Student attendance percentage |
| Parental_Involvement | Level of parental involvement |
| Access_to_Resources | Access to educational resources |
| Extracurricular_Activities | Participation in extracurricular activities |
| Sleep_Hours | Average sleep hours |
| Previous_Scores | Previous academic scores |
| Motivation_Level | Student motivation level |
| Internet_Access | Availability of internet access |
| Tutoring_Sessions | Number of tutoring sessions |
| Family_Income | Family income category |
| Teacher_Quality | Teacher quality |
| School_Type | Type of school |
| Peer_Influence | Influence of peers |
| Physical_Activity | Physical activity hours |
| Learning_Disabilities | Presence of learning disabilities |
| Parental_Education_Level | Parent's education level |
| Distance_from_Home | Distance from home to school |
| Gender | Student gender |
| Exam_Score | Student's exam score |

---

## 🔍 Exploratory Data Analysis

The dataset was analyzed to understand its structure and identify potential issues.

The analysis included:

- Dataset shape
- Data types
- Missing-value analysis
- Duplicate-value checking
- Descriptive statistics
- Exam score distribution
- Outlier analysis
- Categorical feature analysis
- Numerical feature analysis

### Data Quality

There were **no duplicate rows** in the dataset.

Some missing values were present in:

- `Teacher_Quality`
- `Parental_Education_Level`
- `Distance_from_Home`

These values were handled during preprocessing.

---

## 🎯 Performance Categories

The `Exam_Score` column was converted into a categorical target variable called:

```text
Performance_Level
## 🖥️ Application Workflow

```text
User Input
    ↓
DataFrame Creation
    ↓
One-Hot Encoding
    ↓
Feature Alignment
    ↓
Feature Scaling
    ↓
Saved Machine Learning Model
    ↓
Prediction
    ↓
Performance Level
    ↓
Prediction Probabilities
```
## 📁 Project Structure

```text
Student-Performance-Prediction/
│
├── data/
│   └── student_performance.csv
│
├── models/
│   ├── feature_names.pkl
│   ├── scaler.pkl
│   └── student_performance_model.pkl
│
├── notebooks/
│   └── Student_Performance_Analysis.ipynb
│
├── report/
│   └── ...
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```
## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Jupyter Notebook
- Git
- GitHub
- 
