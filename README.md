# 🎓 Academic Performance Analytics

An interactive machine-learning web application for analyzing, predicting, and monitoring student academic performance.

## 🌐 Live Demo

Deployed using Render.

> Add your live Render URL here.

## 📌 Project Overview

Academic Performance Analytics analyzes student academic data and provides:

- 📊 Academic performance dashboard
- 👨‍🎓 Student record management
- 🤖 ML-based GPA prediction
- 🎯 Grade Class prediction
- 🧠 AI Academic Advisor
- ⚠️ At-Risk Student Detection
- 🔬 Model comparison and evaluation
- 🔄 What-If scenario analysis
- ♻️ Model retraining
- 📥 Student data management

The application is designed to help understand academic performance patterns and identify students who may need additional academic attention.

## ✨ Features

### 📊 Dashboard

Provides an overview of the student dataset including:

- Total students
- Average GPA
- Highest GPA
- Lowest GPA
- GPA distribution
- Grade Class distribution
- Study time analysis
- Absence analysis
- Parental support analysis
- Activity analysis

### 🤖 Student Performance Prediction

The prediction system accepts behavioral and demographic student information and predicts:

- Predicted GPA
- Predicted Grade Class
- Performance category
- Prediction probabilities

Actual GPA is not used as an input during Grade Class prediction.

### 🧠 AI Academic Advisor

Provides model-based academic guidance using factors such as:

- Study time
- Absences
- Tutoring
- Parental support
- Extracurricular activities
- Predicted GPA

The recommendations are intended as academic guidance and are not causal conclusions.

### ⚠️ At-Risk Student Detection

The system screens students using ML-based predictions and identifies:

- High-risk students
- Medium-risk students
- Low-risk students

Each flagged student includes:

- Student ID
- Predicted GPA
- Predicted Grade Class
- Risk level
- Risk score
- Risk indicators

The risk system is a screening mechanism and should not be treated as a definitive judgment about a student.

### 🔬 Model Lab

The application provides model evaluation and comparison.

Classification models:

- Logistic Regression
- Random Forest
- Extra Trees
- Gradient Boosting

Regression models:

- Linear Regression
- Random Forest Regressor
- Extra Trees Regressor
- Gradient Boosting Regressor

Evaluation includes:

- Accuracy
- Balanced Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- MAE
- MSE
- RMSE
- R²

### 🔄 What-If Analysis

Allows users to compare a baseline student profile with a modified scenario and observe how the trained model's predictions change.

This is model-based analysis and should not be interpreted as causal evidence.

### ♻️ Model Retraining

The application can retrain the models using the latest dataset.

Before training, the system validates:

- Required columns
- Missing values
- Dataset size
- ML features
- Leakage protection

## 🔐 Machine Learning Design

The project follows two important data-leakage rules:

### StudentID

`StudentID` is used only as an identifier.

It is NOT used as an ML feature.

### GPA

Actual GPA is NOT used as an input to predict Grade Class.

The Grade Class classifier uses only behavioral and demographic features.

## 🧮 ML Features

The classification and GPA prediction models use:

```text
Age
Gender
Ethnicity
ParentalEducation
StudyTimeWeekly
Absences
Tutoring
ParentalSupport
Extracurricular
Sports
Music
Volunteering