import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

# Load model
model = joblib.load("house_price_model.pkl")

st.set_page_config(page_title="AI House Price Estimator", layout="wide")

st.title("🏠 AI House Price Estimator")
st.write("Predict house prices using Machine Learning")

st.sidebar.header("Enter House Details")

area = st.sidebar.number_input("Area (sqft)", 500, 10000, 2000)
bedrooms = st.sidebar.number_input("Bedrooms", 1, 10, 3)
bathrooms = st.sidebar.number_input("Bathrooms", 1, 5, 2)
stories = st.sidebar.number_input("Stories", 1, 4, 2)
parking = st.sidebar.number_input("Parking Spaces", 0, 5, 1)

mainroad = st.sidebar.selectbox("Main Road Access", ["Yes", "No"])
guestroom = st.sidebar.selectbox("Guest Room", ["Yes", "No"])
basement = st.sidebar.selectbox("Basement", ["Yes", "No"])
airconditioning = st.sidebar.selectbox("Air Conditioning", ["Yes", "No"])
prefarea = st.sidebar.selectbox("Preferred Area", ["Yes", "No"])
furnishing = st.sidebar.selectbox("Furnishing", ["Furnished", "Semi-Furnished", "Unfurnished"])

# Convert inputs
input_data = pd.DataFrame({
    "area":[area],
    "bedrooms":[bedrooms],
    "bathrooms":[bathrooms],
    "stories":[stories],
    "parking":[parking],
    "mainroad_yes":[1 if mainroad=="Yes" else 0],
    "guestroom_yes":[1 if guestroom=="Yes" else 0],
    "basement_yes":[1 if basement=="Yes" else 0],
    "hotwaterheating_yes":[0],
    "airconditioning_yes":[1 if airconditioning=="Yes" else 0],
    "prefarea_yes":[1 if prefarea=="Yes" else 0],
    "furnishingstatus_semi-furnished":[1 if furnishing=="Semi-Furnished" else 0],
    "furnishingstatus_unfurnished":[1 if furnishing=="Unfurnished" else 0]
})

if st.sidebar.button("Predict Price", key="predict_btn"):

    price = model.predict(input_data)

    st.subheader("💰 Estimated House Price")
    st.success(f"${price[0]:,.2f}")

    st.write("### Property Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric("Area", f"{area} sqft")
    col2.metric("Bedrooms", bedrooms)
    col3.metric("Bathrooms", bathrooms)

    st.info("This price is predicted using a Machine Learning model.")

if st.sidebar.button("Predict Price"):

    price = model.predict(input_data)

    st.subheader("💰 Estimated House Price")
    st.success(f"${price[0]:,.2f}")

    # -------------------------
    # Dynamic Price Trend Graph
    # -------------------------

    st.subheader("📊 Predicted Price Trend")

    import numpy as np
    import matplotlib.pyplot as plt

    years = np.array([2019,2020,2021,2022,2023])

    base_price = price[0]

    prices = [
        base_price * 0.7,
        base_price * 0.8,
        base_price * 0.9,
        base_price * 0.95,
        base_price
    ]

    fig, ax = plt.subplots()

    ax.plot(years, prices, marker="o")

    ax.set_title("Predicted Price Growth")
    ax.set_xlabel("Year")
    ax.set_ylabel("Price")

    st.pyplot(fig)

#Add House Image Gallery
st.subheader("🏡 Example Houses")

col1, col2, col3 = st.columns(3)

with col1:
    st.image("https://images.unsplash.com/photo-1568605114967-8130f3a36994", caption="Modern House")

with col2:
    st.image("https://images.unsplash.com/photo-1572120360610-d971b9d7767c", caption="Luxury Home")

with col3:
    st.image("https://images.unsplash.com/photo-1507089947368-19c1da9775ae", caption="Family House")

#Add Interactive Map
st.subheader("📍 Property Location")

import pandas as pd

map_data = pd.DataFrame({
    "lat":[22.3039],
    "lon":[70.8022]
})

st.map(map_data)

#Add Feature Importance Chart
st.subheader("📈 Feature Contribution")

features = ["Area","Bedrooms","Bathrooms","Stories","Parking"]

values = [
area,
bedrooms,
bathrooms,
stories,
parking
]

fig_features, ax = plt.subplots()

ax.barh(features, values)

ax.set_title("User Property Features")

st.pyplot(fig_features)