import streamlit as st
import pandas as pd
import pickle

# Load model
model = pickle.load(open("rent_model.pkl", "rb"))

st.title("🏠 House Rent Prediction App")

area = st.number_input("Enter Area (in Sqft)", min_value=0)
bhk = st.selectbox("BHK", [1, 2, 3, 4, 5])
bathroom = st.selectbox("Number of Bathrooms", [1, 2, 3, 4])
city = st.selectbox("City", ["Mumbai", "Bangalore", "Delhi", "Chennai", "Hyderabad"])
furnishing = st.selectbox("Furnishing Status", ["Furnished", "Semi-Furnished", "Unfurnished"])

if st.button("Predict Rent"):
    data = pd.DataFrame([[area, bhk, bathroom, city, furnishing]],
                        columns=["Area", "BHK", "Bathroom", "City", "Furnishing"])
    prediction = model.predict(data)[0]
    st.success(f"Estimated Monthly Rent: ₹ {int(prediction):,}")

