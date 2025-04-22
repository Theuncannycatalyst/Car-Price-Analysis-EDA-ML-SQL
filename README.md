# Car Price Prediction and Analysis

## Overview  
This project tackles a real-world machine learning problem: predicting car prices based on key features such as brand, model year, mileage, and engine size. The workflow spans data cleaning, exploratory data analysis (EDA), model building, evaluation, and delivering actionable insights using Python and SQL.

## Skills Demonstrated  
- **Python**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn  
- **Data Visualization**: EDA, pattern discovery, correlation mapping  
- **Machine Learning**: Regression models (Random Forest, Linear Regression, XGBoost)  
- **SQL**: For querying and slicing the dataset  
- **Feature Engineering**: Cleaning, encoding, scaling  
- **Tools**: Google Colab, GitHub, Jupyter Notebook  

## Dataset  
- 10,000+ used car records (synthetic or anonymized public dataset)  
- Features include: Brand, Model, Year, Mileage, Engine Size, Transmission, and Price  
> **Note**: Dataset shared within notebook.  

## Exploratory Data Analysis (EDA)  
- Identified missing values and handled outliers  
- Explored price trends by brand, year, and mileage  
- Created visualizations: boxplots, bar charts, scatter plots, correlation heatmaps  
- Discovered strong negative correlation between mileage and price, and a strong positive correlation with engine size and recent model years  

## Machine Learning Modeling  
**Model Used**: Random Forest Regressor  
- **MAE (Mean Absolute Error)**: $439.39  
- **R² Score**: 0.9677 → High prediction accuracy  
- Compared performance with Linear Regression and XGBoost  

### Key Steps:  
- One-hot encoding of categorical variables  
- Feature scaling and standardization  
- Train-test split and cross-validation  
- Performance evaluation using MAE, R² Score, and scatter plots  

## Key Insights  
- The model predicted prices with **96.77% accuracy** on test data.  
- Most influential features: **Year, Mileage, and Brand**  
- Dealerships and resellers can optimize pricing and avoid undervaluation or overpricing using this model.  
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
