 import pandas as pd 
 from sklearn.processing import Min MaxScaler 

 data={
    "age": [25,30, 35, 40, 45],
    "height":[150, 160, 170, 180, 190],
    "weight":[50, 60, 70, 80, 90],
 }

 print("Original DtaFrame:")
 print(df)

 scaler = MinMaxScaler
 normalized_data = scaler.fit_transform(df)
normalized_df = pd.DtaFrame(normalized_data, columns=df.columns)
print("\nNormalized DataFrame (scaled to range [0, 1]):")
print(normalized_df)

#assignment
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv("ecommerce_orders.csv")

# Identify missing data
def missing_values_summary(df):
    missing_summary = df.isna().sum()
    missing_percentage = (missing_summary / len(df)) * 100
    return pd.DataFrame({'Missing Values': missing_summary, 'Percentage': missing_percentage})

print(missing_values_summary(df))

# Visualizing missing data patterns
plt.figure(figsize=(10, 6))
sns.heatmap(df.isna(), cmap='viridis', cbar=False, yticklabels=False)
plt.title("Missing Data Heatmap")
plt.show()

# Handling missing values
# Mean/Median imputation for numerical columns (e.g., Product_Price)
numerical_imputer = SimpleImputer(strategy='median')
df['Product_Price'] = numerical_imputer.fit_transform(df[['Product_Price']])

# Mode imputation for categorical columns (e.g., Product_Category)
categorical_imputer = SimpleImputer(strategy='most_frequent')
df['Product_Category'] = categorical_imputer.fit_transform(df[['Product_Category']])

# Forward fill for date-related fields
df['Order_Date'] = df['Order_Date'].fillna(method='ffill')

# KNN imputation for complex cases
knn_imputer = KNNImputer(n_neighbors=5)
df[['Customer_Age']] = knn_imputer.fit_transform(df[['Customer_Age']])

# Evaluate impact
print("Summary statistics before and after imputation:")
print(df.describe())

# Visualizing imputed values
plt.figure(figsize=(10, 5))
sns.boxplot(x=df['Product_Price'])
plt.title("Product Price Distribution After Imputation")
plt.show()

# Save the cleaned dataset
df.to_csv("cleaned_ecommerce_orders.csv", index=False)

print("Data cleaning complete. Cleaned dataset saved as 'cleaned_ecommerce_orders.csv'.")
