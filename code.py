import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
data = pd.read_csv("house_data.csv")

# Display dataset
print(data.head())

# Handling missing values
data = data.dropna()

# Feature selection
X = data[['area', 'bedrooms', 'bathrooms']]
y = data['price']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model creation
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Predict new data
new_house = [[2000, 3, 2]]
predicted_price = model.predict(new_house)
print("Predicted Price:", predicted_price)
