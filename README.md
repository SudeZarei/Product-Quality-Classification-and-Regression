# Product Quality Classification and Regression

This repository contains the code and detailed report for a comprehensive Machine Learning pipeline aimed at predicting wine quality. The project explores both regression and classification approaches to model the physicochemical properties of red wine against sensory quality ratings.

## Project Overview

Wine quality assessment is a critical and expensive process in the wine industry, traditionally relying on physicochemical and sensory tests. This project investigates whether it's possible to predict wine quality objectively using machine learning, answering two core questions:

1. **Classification:** Can we categorize wine into 'Bad', 'Medium', or 'Good' quality?
2. **Regression:** Can we predict the exact numerical quality score (0-10) of a wine?

The dataset used is the Red Wine Quality dataset from the UC Irvine Machine Learning Repository, containing 1599 instances with 11 physicochemical features (e.g., acidity, sugar, alcohol) and one target variable (quality).

## Repository Structure

The project is modularized into distinct directories for clarity and reusability:

- `Models/`: Contains the implementation of various machine learning algorithms.
  - `Classification/`: Code for Logistic Regression, Decision Tree, Random Forest, SVC, and Gradient Boosting. Includes the preprocessed datasets.
  - `Regression/`: Code for Linear Regression, Decision Tree, Random Forest, SVR, and Gradient Boosting. Includes the preprocessed datasets.
- `Preprocessing/`: Scripts used to clean, discretize, encode, and scale the raw data for both tasks.
- `Statistical_Analysis/`: Scripts for exploratory data analysis (EDA), generating boxplots, histograms, and correlation heatmaps.
- `Comparison/`: Scripts dedicated to evaluating and comparing model performances across various metrics.
- `Report.pdf`: The comprehensive project report detailing methodology, analysis, and conclusions.

## Key Findings

Based on the evaluation of 10 different models (5 for regression, 5 for classification), the **Random Forest** algorithm emerged as the top performer for both tasks due to its ability to handle complex, non-linear relationships and feature interactions.

- **Best Classification Model:** Random Forest Classifier
  - _Accuracy:_ ~87.2%
  - _F1-Score:_ ~85.2%
  - _Note:_ Excelled in predicting 'Good' and 'Medium' classes but struggled with the 'Bad' class due to severe dataset imbalance.
- **Best Regression Model:** Random Forest Regressor
  - _MSE:_ 0.3014
  - _R² Score:_ 0.5389

## Exploratory Data Analysis (EDA) Highlights

The statistical analysis revealed several important relationships:

- **Alcohol** has the strongest positive correlation with wine quality.
- **Volatile acidity** (vinegar taste) has a moderate negative correlation with quality.
- **Sulphates** (preservatives) show a positive trend with higher quality scores.

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/SudeZarei/Product-Quality-Classification-and-Regression.git
    cd Product-Quality-Classification-and-Regression
    ```
2.  **Install dependencies:** (Make sure you have a `requirements.txt` file)
    ```bash
    pip install -r requirements.txt
    ```
3.  **Explore the modules:** Navigate to specific folders (`Preprocessing`, `Models`, etc.) to run individual scripts.

## Documentation

For a deep dive into the statistical analysis, preprocessing pipelines, model configurations, and detailed evaluation metrics, please refer to the attached `Report.pdf` in the root directory.

---

_This project was developed by Sude Zarei as a Machine Learning Final Project._
