# Car Price Analysis And Prediction Project

## Overview
This project focuses on analyzing a dataset of **10,000 cars** to understand the factors influencing car prices and building a machine learning model to predict car prices based on features such as brand, model, year, engine size, fuel type, mileage, and more. The project involves **Exploratory Data Analysis (EDA)** and the development of a **Random Forest Regressor** to predict car prices.

---

## Dataset
The dataset contains **10,000 entries**, with each row representing a car and its features. The columns include:
- **Brand**: The manufacturer of the car (e.g., Toyota, Honda).
- **Model**: The specific model of the car (e.g., Camry, Civic).
- **Year**: The year the car was manufactured.
- **Engine_Size**: The engine size in liters.
- **Fuel_Type**: The type of fuel the car uses (e.g., Petrol, Diesel, Electric).
- **Transmission**: The transmission type (e.g., Manual, Automatic).
- **Mileage**: The total distance the car has been driven (in miles).
- **Doors**: The number of doors on the car.
- **Owner_Count**: The number of previous owners.
- **Price**: The price of the car (target variable).

---

## Project Goals
1. **Exploratory Data Analysis (EDA)**:
   - Understand the distribution of key features (e.g., age, mileage, engine size).
   - Identify trends and relationships between features and car prices.
   - Visualize insights to guide decision-making.

2. **Machine Learning**:
   - Build a predictive model to estimate car prices based on features.
   - Evaluate model performance using metrics like **Mean Absolute Error (MAE)** and **R-squared**.

---

## Tools & Libraries
- **Python**: Primary programming language.
- **Pandas**: Data manipulation and cleaning.
- **Matplotlib & Seaborn**: Data visualization.
- **Scikit-learn**: Machine learning model development and evaluation.

---

## Methodology

### 1. Data Cleaning & Preparation
- Loaded the dataset into a pandas DataFrame.
- Checked for missing values (none found).
- Converted categorical variables (e.g., `Fuel_Type`, `Transmission`) into numerical format using **one-hot encoding**.
- Created derived features (e.g., `Car_Age` = 2023 - `Year`).

### 2. Exploratory Data Analysis (EDA)
- **Age Distribution**: Analyzed the distribution of car ages to understand the dataset's composition.
- **Mileage vs. Price**: Explored the relationship between mileage and car prices.
- **Fuel Type Analysis**: Compared prices and mileage across different fuel types.
- **Owner Count Impact**: Investigated how the number of previous owners affects car prices.

### 3. Machine Learning
- **Model**: Random Forest Regressor.
- **Features**: Brand, Model, Year, Engine_Size, Fuel_Type, Transmission, Mileage, Doors, Owner_Count.
- **Target Variable**: Price.
- **Evaluation Metrics**:
  - **Mean Absolute Error (MAE)**: 439.39
  - **R-squared**: 0.9677

---

## Key Findings

### 1. **Car Age & Mileage**:
- Newer cars tend to have **lower mileage** and **higher prices**.
- Older cars with **higher mileage** are generally **cheaper**.

### 2. **Fuel Type**:
- **Electric cars** are newer, have lower mileage, and are more expensive.
- **Diesel cars** have larger engines and mid-range prices.

### 3. **Owner Count**:
- Cars with **fewer owners** tend to be **more expensive**.
- Cars with **more owners** tend to have **higher mileage**.

### 4. **Machine Learning Model**:
- The Random Forest Regressor achieved **96.77% accuracy** (R-squared) with an average error of **$439.39** (MAE).

---

## Visualizations
(Add your charts and visualizations here with brief captions.)

---

## How to Use This Repository

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/car-price-prediction.git
cd car-price-prediction
