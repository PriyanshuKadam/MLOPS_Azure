import streamlit as st
import pandas as pd
import json
import urllib.request
import ssl
import os
import streamlit.components.v1 as components

# Allow self-signed certificates (if necessary)
def allowSelfSignedHttps(allowed):
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

allowSelfSignedHttps(True)

# Define your Azure ML endpoint URL and API key
url = 'https://winequality-qthka.centralindia.inference.ml.azure.com/score'  # Replace with your endpoint URL
api_key = 'mngvSwuxMFneTtl1LfGBocE5z89Z9vn1'  # Replace with your Azure ML API key

# Function to send data to the Azure ML endpoint and get predictions
def get_prediction_from_azure(input_data):
    body = str.encode(json.dumps(input_data))
    headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + api_key}
    
    req = urllib.request.Request(url, body, headers)
    
    try:
        response = urllib.request.urlopen(req)
        result = response.read()
        return json.loads(result)
    except urllib.error.HTTPError as error:
        st.error(f"The request failed with status code: {error.code}")
        st.error(error.read().decode("utf8", 'ignore'))
        return None

# Function to load custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load custom CSS
local_css("style.css")

# Load Animation (You can use particles.js or any other animation here)
animation_symbol = "❄"
st.markdown(
    f"""
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    """,
    unsafe_allow_html=True,
)

# App title
st.title("Wine Quality Prediction with Azure ML")

# Allow the user to test the model by inputting feature values
st.sidebar.header("Test the Model")

# Feature inputs with the specified default values
fixed_acidity = st.sidebar.number_input("Fixed Acidity", min_value=0.0, max_value=15.0, value=11.2, step=0.1)
volatile_acidity = st.sidebar.number_input("Volatile Acidity", min_value=0.0, max_value=2.0, value=0.28, step=0.01)
citric_acid = st.sidebar.number_input("Citric Acid", min_value=0.0, max_value=2.0, value=0.56, step=0.01)
residual_sugar = st.sidebar.number_input("Residual Sugar", min_value=0.0, max_value=20.0, value=1.9, step=0.1)
chlorides = st.sidebar.number_input("Chlorides", min_value=0.0, max_value=0.2, value=0.075, step=0.001)
free_sulfur_dioxide = st.sidebar.number_input("Free Sulfur Dioxide", min_value=0.0, max_value=100.0, value=17.0, step=1.0)
total_sulfur_dioxide = st.sidebar.number_input("Total Sulfur Dioxide", min_value=0.0, max_value=300.0, value=60.0, step=1.0)
density = st.sidebar.number_input("Density", min_value=0.99, max_value=1.05, value=0.998, step=0.0001)
pH = st.sidebar.number_input("pH", min_value=2.5, max_value=4.0, value=3.16, step=0.01)
sulphates = st.sidebar.number_input("Sulphates", min_value=0.0, max_value=2.0, value=0.58, step=0.01)
alcohol = st.sidebar.number_input("Alcohol", min_value=8.0, max_value=15.0, value=9.8, step=0.1)

# Create input data in the format expected by Azure ML
input_data = {
    "input_data": {
        "columns": [
            "fixed acidity",
            "volatile acidity",
            "citric acid",
            "residual sugar",
            "chlorides",
            "free sulfur dioxide",
            "total sulfur dioxide",
            "density",
            "pH",
            "sulphates",
            "alcohol"
        ],
        "index": [0],
        "data": [[
            float(fixed_acidity), float(volatile_acidity), float(citric_acid), float(residual_sugar),
            float(chlorides), float(free_sulfur_dioxide), float(total_sulfur_dioxide), float(density),
            float(pH), float(sulphates), float(alcohol)
        ]]
    }
}

with open("particle.html", "r") as f:
    particle_html = f.read()

# Embed the Particles.js animation in the background
components.html(particle_html, height=250, scrolling=False)

# Make prediction using the Azure ML endpoint
if st.sidebar.button("Predict Wine Quality"):
    st.write("### Sending data to Azure ML for prediction...")
    result = get_prediction_from_azure(input_data)

    # Check if result is valid and display the prediction
    if result:
        # Directly access the first prediction from the list
        prediction = result[0]
        if prediction == True:
            st.success("This wine is predicted to be of **Good Quality**!")
        else:
            st.error("This wine is predicted to be of **Bad Quality**.")
