🌍World Top Company Analysis & AI Dashboard

## Overview
This project is a financial analytics platform that uses machine learning to identify high-growth companies from global market data. It combines a predictive model with an interactive web dashboard, allowing users to visualize capital flow and understand the logic behind the AI's predictions through automated reports.

## Key Features

- **Growth Prediction:** Uses Random Forest and XGBoost models to classify companies as "High Growth" with high accuracy.
- **Automated Analyst Reports:** A natural language generation module creates text summaries explaining exactly why a specific stock was selected (e.g., based on earnings, P/E ratio, or profit margins).
- **Interactive Dashboard:** A full-stack web application built with Dash that features a 3D globe for visualizing market data geographically.
- **What-If Simulator:** Allows users to tweak financial parameters in real-time to see how changes in metrics like profit margin affect the AI's prediction.
- **Model Explainability:** Integrates SHAP values to provide transparency, showing exactly which features influenced the model's decision.

## Tech Stack

- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn (Random Forest, K-Means, Isolation Forest), XGBoost
- **Explainability:** SHAP
- **Visualization:** Plotly, Dash
- **Model Persistence:** Joblib

## Project Structure

- `ml.py`: The main script containing the machine learning pipeline and the report generator.
- `app_globe.py`: The code for the interactive web dashboard.
- `World_Top_Companies_Master_Dataset.csv`: The dataset used for training and analysis.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/MayankRane007/world-top-company-analysis.git](https://github.com/MayankRane007/world-top-company-analysis.git)

Install the required dependencies:
Bash
pip install -r requirements.txt

Run the Analysis Engine (for reports and predictions):
Bash
python ml.py

Launch the Web Dashboard:
Bash
python app_globe.py
