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