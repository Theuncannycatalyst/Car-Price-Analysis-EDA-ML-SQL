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

### 2. Exploratory Data Analysis (EDA)
### 1. **Brand Analysis**
- **Popularity by Brand**: Identified the most common car brands in the dataset using bar charts.
  - *Insight*: Ford and Audi dominated the dataset, reflecting their market prevalence.

- **Price by Brand**: Compared average prices across brands using bar charts and box plots.
  - *Insight*: The domestic brand Chevrolet and the Luxury brand Mercedes had significantly higher average prices compared to budget brands.

- **Top Models**: Analyzed the most popular car models within each brand.
  - *Insight*: Models like Honda Accord and Ford Fiesta were among the top sellers.

### 2. **Fuel Type Analysis**
- **Distribution**: Visualized the proportion of cars by fuel type (Petrol, Diesel, Electric) using pie charts.
  - *Insight*: Petrol cars were the most common (60%), followed by Diesel (25%) and Electric (15%).


- **Price Impact**: Compared average prices across fuel types using bar charts.
  - *Insight*: Electric cars were the most expensive, likely due to newer technology and lower mileage.

### 3. **Mileage and Age**
- **Mileage vs. Year**: Explored the relationship between car age and mileage using scatter plots.
  - *Insight*: Newer cars generally had lower mileage, while older cars showed higher usage.
- **Price Correlation**: Analyzed how mileage impacts price using scatter plots and heatmaps.
  - *Insight*: Higher mileage correlated with lower prices, especially for non-luxury brands.
 
### 4. **Engine Size**
- **Distribution**: Visualized the distribution of engine sizes using histograms.
  - *Insight*: Most cars had engine sizes between 1.5L and 3.0L.
- **Price Impact**: Linked engine size to price using scatter plots.
  - *Insight*: Larger engines (e.g., 4.0L+) were associated with higher prices, particularly in luxury brands.

### 5. **Transmission Type**
- **Price Comparison**: Compared average prices across transmission types (Manual, Automatic, Semi-Automatic).
  - *Insight*: Automatic cars were more expensive on average, likely due to consumer preference.

### 6. **Owner Count**
- **Impact on Price**: Analyzed how the number of previous owners affects price using bar charts.
  - *Insight*: Cars with fewer owners (1-2) commanded higher prices, reflecting buyer preference for lower ownership history.

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

*   **Negative Correlation:** There's a strong negative correlation between car age and mileage. Newer cars generally have lower mileage, reflecting less time on the road. Conversely, older cars typically accumulate higher mileage.
*   **Price Impact:** Car age and mileage are both significant factors in determining price. Newer cars with lower mileage command higher prices, while older cars with higher mileage are generally cheaper. This aligns with common market expectations.

### 2. **Fuel Type**:

*   **Electric Cars:** Electric cars in the dataset tend to be newer, have lower mileage, and command higher prices. This likely reflects the relative newness of electric vehicle technology and potentially government incentives or higher initial purchase costs.
*   **Diesel Cars:** Diesel cars are often associated with larger engine sizes and fall within a mid-range price bracket. This might be due to a combination of factors, such as their use for towing or commercial purposes (requiring larger engines) and potentially a balance between fuel efficiency and purchase cost.

###  3. **Owner Count**:

*   **Price Impact:** Cars with fewer previous owners tend to be more expensive. This is likely due to the perception that fewer owners indicate better care and less wear and tear.
*   **Mileage Association:** Cars with more previous owners tend to have higher mileage. This could be because cars that have been driven more are more likely to change hands more frequently.

### 4. **Machine Learning Model**:
*   **Model Selection:**  A Random Forest Regressor model was chosen for price prediction due to its ability to handle non-linear relationships, its robustness to outliers with high accuracy
*   **Data Splitting:** The data was split into training (80%) and testing (20%) sets using a random seed for reproducibility.
*   **Model Training:** The Random Forest Regressor model was trained on the training data.
*   **Model Evaluation:** The model's performance was evaluated on the test set using the following metrics:
    *   Mean Absolute Error (MAE)
    *   R-squared (R²)
*   **Prediction on New Data:** The trained model was used to predict prices for new, unseen car data.
*   The Random Forest Regressor achieved **96.77% accuracy** (R-squared) with an average error of **$439.39** (MAE).

---

## Key Visualizations
Car Brand Popularity: This chart displays the top 10 most popular car brands based on the number of cars, highlighting Ford's leading position.
![Most popular car brands bar chart](https://github.com/user-attachments/assets/96134e09-f1da-4df3-a9fa-11f0953a9d03)
Average Car Prices:  A comparison of average prices for the top 10 most popular car brands, showing relatively similar price points across brands.
![Averafe price by brand bar chart](https://github.com/user-attachments/assets/d8320418-0e51-4a1b-915c-33802e7b5d1f)
Car Price Variability: Box plots showing the distribution of prices for different car brands, highlighting median prices, interquartile ranges, and outliers.
![Price distribution by brand boxplot chart](https://github.com/user-attachments/assets/288a3da0-31cd-4246-a992-d5c2b440daa8)
Car Brand Fuel Mix: A visualization of car brand popularity, broken down by fuel type, revealing the dominance of petrol and the brand-specific adoption of diesel, electric, and hybrid options
![Brand popularity by fuel type stacked bar chart](https://github.com/user-attachments/assets/264967f7-932e-45ad-bf3f-b68f46b06a0f)
Most Popular Fuel: This chart visualizes the distribution of car counts across different fuel types, highlighting the significant lead of electric vehicles.
![Most popular fuel type bar chart](https://github.com/user-attachments/assets/2197069f-4735-4b1c-a648-106c0c4efb1b)
Car Fuel Type Distribution: A pie chart visualizing the proportion of cars for each fuel type, showing a near-even split among electric, diesel, hybrid, and petrol vehicles.
![Most popular fuel type pie chart](https://github.com/user-attachments/assets/3a603d6a-7416-46c0-8a76-7d01875b6048)
Transmission Type and Price: A bar chart comparing the average prices of cars with different transmission types, highlighting the higher cost of automatic transmissions.
![Average price by transmission bar chart](https://github.com/user-attachments/assets/3f2e8013-943a-40ce-94ae-4584f36b01cc)
Distribution of Car Years: A histogram showing the frequency of cars by year of manufacture, highlighting a higher concentration of newer models.
![Distribution of car years histogram](https://github.com/user-attachments/assets/010ae40c-f2d0-417c-b174-c14013f408b8)
Rising Car Prices: A line chart depicting the average car price trend over the years, illustrating the increasing cost of vehicles over time.
![Average car price by year line chart](https://github.com/user-attachments/assets/64347d0f-bbe6-4810-83bc-54be08a9c962)
Distribution of Car Mileage: A histogram showing the frequency of cars across different mileage levels, revealing a relatively uniform spread.
![Distribution of car mileage histogram](https://github.com/user-attachments/assets/436af819-541f-44b5-a706-6d7a4c01d532)
Mileage Distribution by Year: This chart visualizes the mileage of cars across different years, highlighting the lack of a linear relationship and the wide variation in mileage within each year.
![Mileage vs Year scatter plot](https://github.com/user-attachments/assets/85cf8a23-13b0-4b50-b6ca-85f0930a446d)
Relationship Between Mileage and Price: A scatter plot visualizing the inverse relationship between mileage and price, demonstrating the general trend of decreasing car prices with increasing mileage.
![Mileage vs Price scatterplot](https://github.com/user-attachments/assets/11d03920-819d-447e-a7d2-514214e7c854)
Mileage as Primary Price Factor: The heatmap confirms that mileage is the strongest factor influencing price, showing a strong negative correlation. While year also has a positive influence on price, its relationship with mileage is negligible.
![Year, mileage and price corelation heatmap](https://github.com/user-attachments/assets/4239cc94-5a7d-4bb2-b2e7-9724c08f9073)
Electric Vehicle Premium: The chart reveals a significant price difference based on fuel type, with electric cars commanding a considerably higher average price compared to hybrid, diesel, or petrol options.
![Average price by fuel](https://github.com/user-attachments/assets/aca60efc-e616-423e-87af-d6de5b06e3a8)
Distribution of Engine Sizes: A histogram showing the frequency of cars across different engine size ranges, revealing a relatively uniform spread.
![Distribution of engine size histogram](https://github.com/user-attachments/assets/c48d6e31-a549-4f44-8bfb-c24415d3a8c5)
Engine Size and Price Correlation: This chart illustrates the relationship between engine size and price, highlighting the general trend of increasing price with engine size, but also the significant influence of other factors on price.
![Engine size vs price scatter plot](https://github.com/user-attachments/assets/37d135ce-14e4-4d0f-ab80-c16572bb7fcc)
Diesel and Hybrid Larger Engines: The box plots suggest that Diesel and Hybrid cars tend to have larger engine sizes on average, while Electric vehicles show a narrower range and a lower median engine size. Petrol cars exhibit a wide range but with the lowest median engine size.
![Engine size by fuel type boxplot chart](https://github.com/user-attachments/assets/211824e6-8b89-4a9c-ab96-b096dca35ab3)
Mileage as Primary Price Factor: The heatmap confirms that mileage is a stronger factor influencing price than engine size, showing a moderately strong negative correlation. While engine size also has a positive influence on price, its relationship with mileage is negligible.
![Engine size, price and mileage corelation heatmap](https://github.com/user-attachments/assets/579eb8e5-d1c7-4063-8021-e49185ad701c)
Distribution of Owner Counts: A histogram showing the frequency of cars associated with different numbers of owners, revealing distinct peaks at each whole number count.
![Distribution of owner count histogram](https://github.com/user-attachments/assets/e0ca7617-b973-498d-a92d-0eb9b55e6a6e)
Owner Count Insignificant for Price: The chart demonstrates that the number of previous owners has virtually no impact on the average price of cars in this dataset, suggesting that other factors are much more influential in determining a car's value.
![Average price by owner count bar chart](https://github.com/user-attachments/assets/7a91aa0e-cbc1-44a5-9cf2-363718fe602b)
Fuel Type Insignificant for Owner Count: The box plots demonstrate that the type of fuel a car uses has virtually no impact on the number of previous owners it has had. The distribution of owner counts is remarkably consistent across all fuel types, suggesting that other factors are much more influential in determining how many owners a car has had.
![Owner count by fuel type boxplot chart](https://github.com/user-attachments/assets/933817b4-f4d5-451c-adf5-af00774af7e4)
Strong Predictive Accuracy: The scatter plot reveals a strong linear relationship between actual and predicted car prices, indicating the model's high accuracy in predicting car values.
![Actual vs Predicted Car Prices Machine learning visual](https://github.com/user-attachments/assets/94fc1369-9460-4a59-a6cd-91061c32afb2)
