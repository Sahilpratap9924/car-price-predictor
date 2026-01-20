# 🚗 Car Price Predictor

A machine learning application that predicts car prices based on various features using a Linear Regression model with one-hot encoding preprocessing.

## Overview

This project uses scikit-learn's Pipeline architecture to build a robust ML model that predicts car prices. The application provides a user-friendly interface built with Streamlit for easy interaction and predictions.

## Features

- **Machine Learning Model**: Linear Regression with OneHotEncoder preprocessing
- **Feature Engineering**: Handles categorical features (car name, company, fuel type) and numerical features (year, kilometers driven)
- **Web Interface**: Interactive Streamlit app for real-time predictions
- **Model Persistence**: Serialized model for quick loading and inference
- **Data Preprocessing**: Automated categorical encoding through sklearn Pipeline

## Project Structure

```
car-price-predictor/
├── app.py                          # Main Streamlit web application
├── retrain_and_save_model.py       # Model training and serialization script
├── cleaned_car.csv                 # Dataset with cleaned car data
├── LinearRegressionModel.pkl       # Trained model (binary)
└── README.md                        # This file
```

## Installation

### Prerequisites

- Python 3.7+
- pip (Python package manager)

### Setup

1. Clone or download this project to your local machine
2. Navigate to the project directory:

   ```bash
   cd car-price-predictor
   ```

3. Install required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   Or install manually:

   ```bash
   pip install streamlit pandas scikit-learn
   ```

## Usage

### Running the Web Application

Start the Streamlit app to make price predictions:

```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

**Interface**:

- Select car name, company, year, kilometers driven, and fuel type
- Click predict to get the estimated car price
- Built-in caching for faster load times

### Retraining the Model

To retrain the model with updated data:

```bash
python retrain_and_save_model.py
```

This will:

1. Load the cleaned car dataset
2. Preprocess categorical features using OneHotEncoder
3. Train a new Linear Regression model
4. Save the model as `LinearRegressionModel.pkl`

## Model Details

### Architecture

```
Input Features
    ↓
ColumnTransformer
    ├─ OneHotEncoder (categorical: name, company, fuel_type)
    └─ Passthrough (numerical: year, kms_driven)
    ↓
LinearRegression Model
    ↓
Price Prediction
```

### Input Features

- **name**: Car model name (categorical)
- **company**: Manufacturer (categorical)
- **year**: Year of manufacture (numerical)
- **kms_driven**: Kilometers driven (numerical)
- **fuel_type**: Type of fuel (categorical)

### Output

- **Price**: Predicted car price (numerical)

## Data

The model is trained on `cleaned_car.csv` which contains:

- Car specifications and features
- Historical pricing data
- Pre-cleaned and preprocessed data ready for ML

## Technologies Used

- **Python 3**: Core programming language
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms and preprocessing
- **Pickle**: Model serialization

## How It Works

1. **Data Preparation**: The cleaned car dataset is loaded and split into features and target
2. **Preprocessing**: A ColumnTransformer handles:
   - One-hot encoding for categorical variables
   - Passthrough for numerical variables
3. **Model Training**: Linear Regression fits on preprocessed data
4. **Pipeline Assembly**: Both steps are wrapped in a Pipeline for seamless preprocessing during prediction
5. **Serialization**: The complete pipeline is saved as a pickle file
6. **Inference**: User inputs are automatically preprocessed and passed through the model for predictions

## Future Enhancements

- Add additional features (mileage, condition, etc.)
- Implement advanced models (Random Forest, Gradient Boosting)
- Add model performance metrics and visualization
- Include prediction confidence intervals
- Support for model versioning

## Notes

- The model uses a simple Linear Regression approach; more complex models may provide better accuracy
- Categorical features with unknown values during prediction are handled gracefully by OneHotEncoder
- The Streamlit app caches data and model loading for improved performance
