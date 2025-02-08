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
```
# Load the  dataset
df = pd.read_csv(r"C:\...\Data sets from kaggle\car_price_dataset.csv")

#prints first 5 rows
print(df.head())

#print information about dataframe
print(df.info())

# Check if there are any missing values in the DataFrame
print(df.isnull().values.any())

#print the rows with null values
print(df.isnull().sum())
```

### 2. Exploratory Data Analysis (EDA)
### 1. **Brand Analysis**
- **Popularity by Brand**: Identified the most common car brands in the dataset using bar charts.
  - *Insight*: Ford and Audi dominated the dataset, reflecting their market prevalence.
```
#group by brand and count the number of cars
car_popular = df.groupby(["Brand"]).size().reset_index(name="Count")

#sort by count for better visuals
car_popular = car_popular.sort_values(by="Count", ascending=False)

#print the result
print(car_popular)

#plotting the bar chart
plt.figure(figsize=(12, 6))
plt.bar(car_popular["Brand"], car_popular["Count"], color="skyblue")
plt.title("Most popular car brands")
plt.xlabel("Brand")
plt.ylabel("Number of cars")
plt.xticks(rotation=45)
plt.show()
```
- **Price by Brand**: Compared average prices across brands using bar charts and box plots.
  - *Insight*: The domestic brand Chevrolet and the Luxury brand Mercedes had significantly higher average prices compared to budget brands.
 ```
#group by brand and calculate the average price
brand_price = df.groupby("Brand")["Price"].mean().reset_index(name="Average_Price")

#sort by average price for better visualization
brand_price = brand_price.sort_values(by="Average_Price", ascending=False)

#print the result
print(brand_price)

#plot a bar chart
plt.figure(figsize=(12, 6))
plt.bar(brand_price["Brand"], brand_price["Average_Price"], color = "lightgreen")
plt.title("Average price by brand")
plt.xlabel("Brand")
plt.ylabel("Average Price")
plt.xticks(rotation=45)
plt.show()

#using a box plot to show price range for each car
plt.figure(figsize=(12, 6))
sns.boxplot(x="Brand", y="Price", data=df, palette="Set3")
plt.title("Price Distribution by Brand")
plt.xlabel("Brand")
plt.ylabel("Price")
plt.xticks(rotation=45)
plt.show()
```
- **Top Models**: Analyzed the most popular car models within each brand.
  - *Insight*: Models like Honda Accord and Ford Fiesta were among the top sellers.
```
#group by brand and model
model_popular = df.groupby(["Brand", "Model"]).size().reset_index(name="Count")

#sort by count for better visuals
model_popular = model_popular.sort_values(by="Count", ascending=False)

#print the result
print(model_popular.head(10))

#plot the top 10 models
plt.figure(figsize=(12, 6))
sns.barplot(x="Count", y="Model", hue="Brand", data=model_popular.head(10), palette="viridis")
plt.title("Top 10 Most Popular Car Models by Brands")
plt.xlabel("Number of Cars")
plt.ylabel("Model")
plt.show()
```

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
