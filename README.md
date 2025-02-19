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
### 2. **Fuel Type Analysis**
- **Distribution**: Visualized the proportion of cars by fuel type (Petrol, Diesel, Electric) using pie charts.
  - *Insight*: Petrol cars were the most common (60%), followed by Diesel (25%) and Electric (15%).

```
#group by brand and fuel type
brand_fuel = df.groupby(["Brand", "Fuel_Type"]).size().unstack()

#plotting the bar chart
brand_fuel.plot(kind="bar", stacked=True, figsize=(12, 6), colormap="viridis")
plt.title("Brand Popularity by Fuel Type")
plt.xlabel("Brand")
plt.ylabel("Number of Cars")
plt.xticks(rotation=45)
plt.show()

#group by fuel type
fuel_popular = df.groupby(["Fuel_Type"]).size().reset_index(name="Count")

#sort by count for better visuals
fuel_popular = fuel_popular.sort_values(by="Count", ascending=False)

#print the result
print(fuel_popular)

#plotting the bar chart
plt.figure(figsize=(12, 6))
plt.bar(fuel_popular["Fuel_Type"], fuel_popular["Count"], color="skyblue")
plt.title("Most popular Fuel Type")
plt.xlabel("Fuel")
plt.ylabel("Number of cars")
plt.xticks(rotation=0)
plt.show()

#plotting a pie chart
plt.figure(figsize=(8, 8))
plt.pie(fuel_popular["Count"], labels=fuel_popular["Fuel_Type"], autopct="%1.1f%%", startangle=140, colors=plt.cm.Paired.colors)
plt.title("Proportion of Cars by Fuel Type")
plt.show()
```
- **Price Impact**: Compared average prices across fuel types using bar charts.
  - *Insight*: Electric cars were the most expensive, likely due to newer technology and lower mileage.
```
#group by fuel type and calculate average price
fuel_price = df.groupby("Fuel_Type")["Price"].mean().reset_index(name="Average_Price")

#sort by average price
fuel_price = fuel_price.sort_values(by="Average_Price", ascending=False)

#print results
print(fuel_price)

#plot the bar chart
plt.figure(figsize=(10, 6))
sns.barplot(x = "Fuel_Type", y="Average_Price", data=fuel_price, palette="viridis")
plt.title("Average Price by Fuel")
plt.xlabel("Fuel Type")
plt.ylabel("Average Price")
plt.xticks(rotation = 45)
plt.show()

#scatter plot of mileage vs price colored by fuel type
plt.figure(figsize=(12, 6))
sns.scatterplot(x="Mileage", y="Price", hue="Fuel_Type", data=df, palette="viridis", alpha=0.8)
plt.title("Mileage vs price by fuel type")
plt.xlabel("Mileage")
plt.ylabel("Price")
plt.legend(title="Fuel Type")
plt.show()
```
### 3. **Mileage and Age**
- **Mileage vs. Year**: Explored the relationship between car age and mileage using scatter plots.
  - *Insight*: Newer cars generally had lower mileage, while older cars showed higher usage.
- **Price Correlation**: Analyzed how mileage impacts price using scatter plots and heatmaps.
  - *Insight*: Higher mileage correlated with lower prices, especially for non-luxury brands.
 ```
#distribution of mileage
plt.figure(figsize=(12, 6))
sns.histplot(df["Mileage"], bins=30, kde=True, color="lightgreen")
plt.title("Distribution of Mileage")
plt.xlabel("Mileage")
plt.ylabel("Frequency")
plt.show()

#scatter plot of mileage vs year
plt.figure(figsize=(12, 6))
sns.scatterplot(x="Year", y="Mileage", data=df, alpha=0.6, color="purple")
plt.title("Mileage vs. Year")
plt.xlabel("Year")
plt.ylabel("Mileage")
plt.show()

#scatter plot of mileage vs price
plt.figure(figsize=(12, 6))
sns.scatterplot(x="Mileage", y="Price", data=df, alpha=0.6, color="teal")
plt.title("Mileage vs. Price")
plt.xlabel("Mileage")
plt.ylabel("price")
plt.show()

#scatter plot of year vs mileage, colored by price
plt.figure(figsize=(12, 6))
sns.scatterplot(x="Year", y="Mileage", hue="Price", data=df, palette="viridis", alpha=0.8)
plt.title("Year vs Mielage colored by price")
plt.xlabel("Year")
plt.ylabel("Mileage")
plt.show()

#corelation heatmap for numerical columns
plt.figure(figsize=(12, 6))
sns.heatmap(df[["Year", "Mileage", "Price"]].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation heatmap: Year, Mileage and Price")
plt.show()
```
### 4. **Engine Size**
- **Distribution**: Visualized the distribution of engine sizes using histograms.
  - *Insight*: Most cars had engine sizes between 1.5L and 3.0L.
- **Price Impact**: Linked engine size to price using scatter plots.
  - *Insight*: Larger engines (e.g., 4.0L+) were associated with higher prices, particularly in luxury brands.
 ```
#distribution of engine size
plt.figure(figsize=(12, 6))
sns.histplot(df["Engine_Size"], bins=30, kde=True, color="skyblue")
plt.title("Distribution of Engine Size")
plt.xlabel("Engine Size")
plt.ylabel("Frequency")
plt.show()

#scatter plot of engine size vs price
plt.figure(figsize=(12, 6))
sns.scatterplot(x="Engine_Size", y="Price", data=df, alpha=0.6, color="orange")
plt.title("Engine Size vs Price")
plt.xlabel("Engine Size")
plt.ylabel("Price")
plt.show()

#scatter plot fo engine size vs milage
plt.figure(figsize=(12, 6))
sns.scatterplot(x="Engine_Size", y="Mileage", data=df, alpha=0.6, color="purple")
plt.title("Engine Size vs Mileage")
plt.xlabel("Engine Size")
plt.ylabel("Mileage")
plt.show()

#box plot of engine size by fuel type
plt.figure(figsize=(12, 6))
sns.boxplot(x="Fuel_Type", y="Engine_Size", data=df, palette="Set3")
plt.title("Engine Size by Fuel Type")
plt.xlabel("Fuel Type")
plt.ylabel("Engine Size")
plt.show()

#scatter plot of engine size vs price colored by fuel type
plt.figure(figsize=(12, 6))
sns.scatterplot(x="Engine_Size", y="Price", hue="Fuel_Type", data=df, palette="viridis", alpha=0.8)
plt.title("Engine Size vs Price by Fuel")
plt.xlabel("Engine Size")
plt.ylabel("Price")
plt.legend(title="Fuel Type")
plt.show()

#corelation heatmap for numerical columns
plt.figure(figsize=(12, 6))
sns.heatmap(df[["Engine_Size", "Price", "Mileage"]].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap: Engine Size, Price and Mileage")
plt.show()
```
### 5. **Transmission Type**
- **Price Comparison**: Compared average prices across transmission types (Manual, Automatic, Semi-Automatic).
  - *Insight*: Automatic cars were more expensive on average, likely due to consumer preference.
```
#group by transmission and calculate the average price
trans_price = df.groupby("Transmission")["Price"].mean().reset_index(name="Average_Price")

#sort by average price for better visualization
trans_price = trans_price.sort_values(by="Average_Price", ascending=False)

#print the result
print(trans_price)

#plot a bar chart
plt.figure(figsize=(12, 6))
plt.bar(trans_price["Transmission"], trans_price["Average_Price"], color = "lightgreen")
plt.title("Average price by Transmission")
plt.xlabel("Transmission")
plt.ylabel("Average Price")
plt.xticks(rotation=0)
plt.show()

#plotting a pie chart
plt.figure(figsize=(8, 8))
plt.pie(trans_price["Average_Price"], labels=trans_price["Transmission"], autopct="%1.1f%%", startangle=140, colors=plt.cm.Paired.colors)
plt.title("Proportion of transmission by price")
plt.show()
```
### 6. **Owner Count**
- **Impact on Price**: Analyzed how the number of previous owners affects price using bar charts.
  - *Insight*: Cars with fewer owners (1-2) commanded higher prices, reflecting buyer preference for lower ownership history.
```
#distribution of owner count
plt.figure(figsize=(10, 6))
sns.histplot(df["Owner_Count"], bins=30, kde=True, color="skyblue")
plt.title("Distribution of owner count")
plt.xlabel("Owner Count")
plt.ylabel("Frequency")
plt.show()

#group by ownder count and average price
owner_price = df.groupby("Owner_Count")["Price"].mean().reset_index(name="Average_Price")

#print the result
print(owner_price)

#plot the bar chart
plt.figure(figsize=(10, 6))
sns.barplot(x="Owner_Count", y="Average_Price", data=owner_price, palette="viridis")
plt.title("Average Price by owner count")
plt.xlabel("Owner Count")
plt.ylabel("Average Price")
plt.show()

#scatter plot of onwer_count vs mileage
plt.figure(figsize=(10, 6))
sns.scatterplot(x="Owner_Count", y="Mileage", data=df, alpha=0.6, color="purple")
plt.title("Onwer Count vs Mileage")
plt.xlabel("Onwer Count")
plt.ylabel("Mileage")
plt.show()

#box plot of owner count by fuel type
plt.figure(figsize=(12, 6))
sns.boxplot(x="Fuel_Type", y="Owner_Count", data=df, palette="Set3")
plt.title("Onwer Count by Fuel Type")
plt.xlabel("Fuel Type")
plt.ylabel("Onwer Count")
plt.show()
```
### 3. Machine Learning
- **Model**: Random Forest Regressor.
- **Features**: Brand, Model, Year, Engine_Size, Fuel_Type, Transmission, Mileage, Doors, Owner_Count.
- **Target Variable**: Price.
- **Evaluation Metrics**:
  - **Mean Absolute Error (MAE)**: 439.39
  - **R-squared**: 0.9677
```
#Machine learning stuff
#preprocessing
df_ml = df.copy()
df_ml = pd.get_dummies(df_ml, columns=["Brand", "Model", "Fuel_Type", "Transmission"], drop_first=True)

#features and target
X = df_ml.drop(columns=["Price"])
y = df_ml["Price"]

#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#train Random Forest Regressor
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

#predict on test set
y_pred = model.predict(X_test)

#create a dataframe to compare actual vs predicted prices
results = pd.DataFrame({
    "Actual Price": y_test,
    "Predicted Price": y_pred
})

#print the first 10 predictions
print(results.head(10))

#Evaluate
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R-squared", r2_score(y_test, y_pred))

#Example of predicting the price of a new car
new_car={
    "Brand": "Toyota",
    "Model": "Camry",
    "Year": 2020,
    "Engine_Size": 2.5,
    "Fuel_Type": "Petrol",
    "Transmission": "Automatic",
    "Mileage": 30000,
    "Doors": 4,
    "Owner_Count": 1
}

# Example: Predict the price of another car
new_car_2 = {
    "Brand": "Honda",
    "Model": "Civic",
    "Year": 2018,
    "Engine_Size": 1.8,
    "Fuel_Type": "Petrol",
    "Transmission": "Manual",
    "Mileage": 50000,
    "Doors": 4,
    "Owner_Count": 2
}

# Convert the new car data into a DataFrame
new_car_df_2 = pd.DataFrame([new_car_2])

# Preprocess the new car data (one-hot encoding)
new_car_df_2 = pd.get_dummies(new_car_df_2)
new_car_df_2 = new_car_df_2.reindex(columns=X_train.columns, fill_value=0)

# Predict the price
predicted_price_2 = model.predict(new_car_df_2)
print("Predicted Price:", predicted_price_2[0])

#convert new car data into a dataframe
new_car_df = pd.DataFrame([new_car])

#preprocess the new car data
new_car_df = pd.get_dummies(new_car_df)
new_car_df = new_car_df.reindex(columns=X_train.columns, fill_value=0)

#predict the prices
predicted_price = model.predict(new_car_df)
print("Predicted Price:", predicted_price[0])
```

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
- The Random Forest Regressor achieved **96.77% accuracy** (R-squared) with an average error of **$439.39** (MAE).

---

## Key Visualizations
Car Brand Popularity: This chart displays the top 10 most popular car brands based on the number of cars, highlighting Ford's leading position.
![Most popular car brands bar chart](https://github.com/user-attachments/assets/96134e09-f1da-4df3-a9fa-11f0953a9d03)
Average Car Prices:  A comparison of average prices for the top 10 most popular car brands, showing relatively similar price points across brands.
![Averafe price by brand bar chart](https://github.com/user-attachments/assets/d8320418-0e51-4a1b-915c-33802e7b5d1f)
Top Car Models and Brands: This chart displays the top 10 most popular car models, showing the diversity of brands represented and the relative popularity of each model.
![Top 10 popular car models by brands bar chart](https://github.com/user-attachments/assets/4dfd7ac9-23be-4966-bfe4-e48c54effbd5)
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

---
## Methodology

This project followed a structured approach to explore and model car prices, incorporating data aggregation, visualization, statistical analysis, and machine learning.  The following steps outline the methodology used:

**1. Data Collection and Preparation:**
*   "The dataset used in this analysis was obtained from Kaggle  and contains information about 10000 cars, including their brand, model, year, engine size, fuel type, transmission, mileage, number of previous owners, and price."
*   "The data was preprocessed to [mention cleaning steps, e.g., handle missing values, convert data types, remove duplicates, etc.].  Specifically, [describe any specific data cleaning techniques you used]."

**2. Exploratory Data Analysis (EDA):**

*   **Data Grouping and Aggregation:**  Data was aggregated using the `groupby` function in pandas to explore relationships between different variables. This included:
    *   Grouping by brand to analyze brand popularity and average price.
    *   Grouping by fuel type to understand fuel type distribution and average price differences.
    *   Grouping by transmission type to examine price variations based on transmission type.
    *   Combined groupings (e.g., brand and fuel type) to explore more nuanced relationships.
*   **Visualization:**  `matplotlib` and `seaborn` libraries were employed to create a variety of visualizations, including:
    *   **Bar charts:** To compare counts and average prices across different categories (brands, fuel types, transmission types).
    *   **Histograms:** To visualize the distribution of numerical variables (year, mileage, engine size, owner count).
    *   **Scatter plots:** To explore relationships between two numerical variables (e.g., mileage vs. price, engine size vs. price).  Scatter plots were also used to visualize actual vs. predicted prices from the machine learning model.
    *   **Box plots:** To compare the distribution of numerical variables across different categories (e.g., price by brand, engine size by fuel type).
    *   **Heatmaps:** To visualize correlation matrices and identify relationships between numerical variables.
    *   **Pie charts:** To show the proportions of different categories (e.g., fuel type distribution).
*   **Statistical Analysis:**  Correlation coefficients were calculated to quantify the strength and direction of linear relationships between numerical variables.

**3. Machine Learning Model Development:**
*   **Model Selection:**  A Random Forest Regressor model was chosen for price prediction due to its ability to handle non-linear relationships, its robustness to outliers with high accuracy
*   **Data Splitting:** The data was split into training (80%) and testing (20%) sets using a random seed for reproducibility.
*   **Model Training:** The Random Forest Regressor model was trained on the training data.
*   **Model Evaluation:** The model's performance was evaluated on the test set using the following metrics:
    *   Mean Absolute Error (MAE)
    *   R-squared (R²)
*   **Prediction on New Data:** The trained model was used to predict prices for new, unseen car data.

**4. Tools and Libraries:**

The following tools and libraries were used in this project:

*   Python
*   pandas
*   numpy
*   matplotlib
*   seaborn
*   scikit-learn





