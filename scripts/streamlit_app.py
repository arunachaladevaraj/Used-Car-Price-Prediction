import streamlit as st
import pickle
import pandas as pd

# Load the trained model
with open("C:\\Users\\arund\\Downloads\\random_forest_model.pkl", 'rb') as file:
    model = pickle.load(file)

# Streamlit app layout
st.markdown(
    """
    <style>
    .centered-title {
        font-size: 50px;
        font-weight: bold;
        text-align: center;
        color: black;
        text-transform: uppercase;
    }

    .price-display {
        font-size: 36px;
        font-weight: bold;
        color: black;
        text-align: center;
        animation: fadeIn 2s ease-in-out;
    }

    @keyframes fadeIn {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }
    </style>
    <div class="centered-title">USED CAR PRICE PREDICTION</div>
    """,
    unsafe_allow_html=True
)

# Sidebar for user input
st.sidebar.header('Input Features')

transmission = st.sidebar.selectbox('Transmission', ['Automatic', 'Manual'])
year_of_manufacture = st.sidebar.selectbox('Year of Manufacture', list(range(2000, 2025)))
model_year = st.sidebar.selectbox('Model Year', list(range(1980, 2025)))
city = st.sidebar.selectbox('City', ['Delhi', 'Banglore', 'Chennai', 'Kolkata', 'Hyderabad', 'Jaipur'])
insurance_validity = st.sidebar.selectbox('Insurance Validity', ['Third Party insurance', 'Zero Dep', 'Comprehensive'])
ownership = st.sidebar.selectbox('Ownership', ['First Owner', 'Second Owner', 'Third Owner', 'Fourth Owner', 'Fifth Owner'])
fuel_type = st.sidebar.selectbox('Fuel Type', ['Cng', 'Petrol', 'Diesel'])
kilometers_driven = st.sidebar.slider('Kilometers Driven', 10000, 200000, step=1000)
body_type = st.sidebar.selectbox('Body Type', ['Hatchback', 'Sedan', 'SUV', 'MUV', 'Minivans', 'Coupe'])
mileage = st.sidebar.slider('Mileage (km/l)', 10, 30, step=1)

# Display image based on selected body type
if body_type == 'Sedan':
    st.image("D:\\python\\car dheko\\sedan.jpg", caption='Sedan', use_column_width=True)
elif body_type == 'SUV':
    st.image("D:\\python\\car dheko\\suv.jpg", caption='SUV', use_column_width=True)
elif body_type == 'Hatchback':
    st.image("D:\\python\\car dheko\\hatchback.jpg", caption='Hatchback', use_column_width=True)
elif body_type == 'MUV':
    st.image("D:\\python\\car dheko\\muv.jpg", caption='MUV', use_column_width=True)
elif body_type == 'Minivans':
    st.image("D:\\python\\car dheko\\minivan.jpg", caption='Minivans', use_column_width=True)
elif body_type == 'Coupe':
    st.image("D:\\python\\car dheko\\coupe.jpg", caption='Coupe', use_column_width=True)

# Predict button
if st.sidebar.button('Predict Price'):
    # Create a DataFrame of input data for the model
    input_data = pd.DataFrame(
        [[transmission, year_of_manufacture, model_year, city, insurance_validity, ownership, fuel_type, kilometers_driven, body_type, mileage]],
        columns=['transmission', 'Year of Manufacture', 'modelYear', 'City', 'Insurance Validity', 'Ownership', 'fuelType', 'kilometersDriven', 'bodyType', 'Mileage']
    )

    # Make prediction using the loaded model
    prediction = model.predict(input_data)
    
    # Display the prediction with animation and center alignment
    st.markdown(
        f"""
        <div class="price-display">Estimated Car Price: ₹{prediction[0]:,.2f}</div>
        """,
        unsafe_allow_html=True
    )
