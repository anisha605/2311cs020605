import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Data
data = {
    "customer_id": [1, 2, 3, 4],
    "gender": ["Male", "Female", "Female", "Male"],  # Corrected typo in "Feamle"
    "city": ["Hyderabad", "Pune", "Banglore", "Mumbai"],  # Standardized case
    "fruits": ["Apple", "Orange", "Kivi", "Banana"]  # Standardized case
}

# Create DataFrame
df = pd.DataFrame(data)

# Initialize OneHotEncoder
one_hot_encoder = OneHotEncoder(sparse_output=False)

# Columns to encode
columns_to_encode = ["gender", "city", "fruits"]

# Fit and transform the data
encoded_data = one_hot_encoder.fit_transform(df[columns_to_encode])

# Get encoded column names
encoded_columns = one_hot_encoder.get_feature_names_out(columns_to_encode)

# Create a new DataFrame with encoded columns
encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns)

# Combine original and encoded data
result_df = pd.concat([df[["customer_id"]], encoded_df], axis=1)

print(result_df)
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv("ecommerce_reviews.csv")

# Handling missing values
# Check missing values
def missing_values_summary(df):
    return df.isnull().sum()

# Impute missing numerical values (e.g., Customer_Age) with median
imputer = SimpleImputer(strategy='median')
df['Customer_Age'] = imputer.fit_transform(df[['Customer_Age']])

# Fill missing Review_Text using NLP (e.g., replace with "No Review" or predictive text completion)
df['Review_Text'].fillna("No Review", inplace=True)

# Handling missing values in Rating (impute with mode)
df['Rating'].fillna(df['Rating'].mode()[0], inplace=True)

# Detect and remove duplicates
df.drop_duplicates(subset=['Review_Text', 'Rating', 'Customer_Age'], keep='first', inplace=True)

# Standardizing Rating values
df = df[(df['Rating'] >= 1) & (df['Rating'] <= 5)]

# Correct spelling inconsistencies in Product_Category
def correct_spelling(text):
    return str(TextBlob(text).correct())

df['Product_Category'] = df['Product_Category'].apply(lambda x: correct_spelling(x) if isinstance(x, str) else x)

# Identifying and handling outliers
# Using boxplot to identify outliers
plt.figure(figsize=(10, 5))
sns.boxplot(x=df['Product_Price'])
plt.show()

# Handling outliers (removing extreme values beyond 1.5*IQR)
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

df = remove_outliers(df, 'Product_Price')

df = remove_outliers(df, 'Rating')

# Convert categorical data into numerical format
label_encoder = LabelEncoder()
df['Product_Category'] = label_encoder.fit_transform(df['Product_Category'])

# Save the cleaned dataset
df.to_csv("cleaned_ecommerce_reviews.csv", index=False)

print("Data cleaning complete. Cleaned dataset saved as 'cleaned_ecommerce_reviews.csv'.")
