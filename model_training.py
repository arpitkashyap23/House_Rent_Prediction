import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

data = pd.DataFrame({
    "Area": [500, 1000, 1500, 2000],
    "BHK": [1, 2, 3, 4],
    "Bathroom": [1, 2, 2, 3],
    "City": ["Mumbai", "Bangalore", "Delhi", "Chennai"],
    "Furnishing": ["Furnished", "Semi-Furnished", "Unfurnished", "Furnished"],
    "Rent": [15000, 25000, 30000, 40000]
})

X = data.drop("Rent", axis=1)
y = data["Rent"]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(), ["City", "Furnishing"])
], remainder="passthrough")

model = Pipeline([
    ("preprocess", preprocessor),
    ("regressor", RandomForestRegressor())
])

model.fit(X, y)

with open("rent_model.pkl", "wb") as f:
    pickle.dump(model, f)

