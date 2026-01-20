import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")

@st.cache_data
def load_data():
    df = pd.read_csv('Cleaned_car.csv')
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    return df

@st.cache_resource
def load_model():
    with open('LinearRegressionModel.pkl', 'rb') as f:
        return pickle.load(f)

car = load_data()
model = load_model()

st.markdown("<h1 style='text-align: center;'>🚗 Car Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: grey;'>ML model using OneHotEncoder + Pipeline</p>", unsafe_allow_html=True)

st.divider()

# ---- UI Inputs ----
col1, col2 = st.columns(2)

with col1:
    name = st.selectbox("Car Name", sorted(car['name'].unique()))
    company = st.selectbox("Company", sorted(car['company'].unique()))
    year = st.number_input("Year", min_value=int(car['year'].min()), max_value=int(car['year'].max()), value=2018)

with col2:
    kms_driven = st.number_input("Kilometers Driven", min_value=0, max_value=500000, value=30000, step=500)
    fuel_type = st.selectbox("Fuel Type", sorted(car['fuel_type'].unique()))

# ---- Input DataFrame (must match training features) ----
input_df = pd.DataFrame({
    'name': [name],
    'company': [company],
    'year': [year],
    'kms_driven': [kms_driven],
    'fuel_type': [fuel_type]
})

st.divider()

# ---- Prediction ----
if st.button("🔮 Predict Price", use_container_width=True):
    try:
        prediction = model.predict(input_df)[0]
        st.markdown(f"<h2 style='text-align: center; color: green;'>₹ {int(prediction):,}</h2>", unsafe_allow_html=True)
        st.success("Prediction Successful!")
    except Exception as e:
        st.error("❌ Prediction failed. Check feature names and model file.")
        st.code(str(e))

st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>Built with ❤️ using Streamlit</p>", unsafe_allow_html=True)
