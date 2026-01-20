import pandas as pd
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# Load your ACTUAL data
df = pd.read_csv("cleaned_car.csv")

if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Features and target (same as your project)
X = df[['name', 'company', 'year', 'kms_driven', 'fuel_type']]
y = df['Price']

# Categorical and numeric columns
cat_cols = ['name', 'company', 'fuel_type']
num_cols = ['year', 'kms_driven']

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ('num', 'passthrough', num_cols)
    ]
)

# Model
model = LinearRegression()


pipe = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('model', model)
])

pipe.fit(X, y)


with open("LinearRegressionModel.pkl", "wb") as f:
    pickle.dump(pipe, f)

print("✅ New compatible model saved as LinearRegressionModel.pkl")
