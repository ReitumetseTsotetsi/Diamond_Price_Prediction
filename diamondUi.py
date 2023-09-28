

import streamlit as st
import pandas as pd
import pickle
from PIL import Image
from sklearn.preprocessing import StandardScaler
import numpy as np


# Define the custom CSS style
custom_style = """
<style>
.frame {
    border: length 1000px, width 800px, white;
    padding: 10px;
}

.title {
    font-family: Arial, sans-serif;
    color: ;
    font-size: 24px;
    font-weight: bold;
}

.label {
    font-size: 16px;
    margin-bottom: 5px;
}

.input {
    padding: 5px;
    border: 1px solid #ccc;
    border-radius: 4px;
    width: 200px;
}

.button {
    background-color: blue;
    color: white;
    padding: 10px 20px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-weight: bold;
}

.button:hover {
    background-color: red;
}
</style>
"""

# Apply the custom style
st.markdown(custom_style, unsafe_allow_html=True)

# Add the title with custom style
st.markdown('<div class="frame"><h1>Diamond Price Prediction App</h1></div>', unsafe_allow_html=True)

st.markdown('<div style="display: flex; justify-content: center;">', unsafe_allow_html=True)
# Add the image
image_url = "diamonds.jpeg"
# Open the image using PIL
image = Image.open(image_url)

# Define the desired width and height
new_width = 600
new_height = 400

# Resize the image
resized_image = image.resize((new_width, new_height))

# Display the resized image
st.image(resized_image, caption="Resized Image", width=new_width)

selected_page = st.sidebar.selectbox("Select Page", ["Home", "Prediction"])

if selected_page == "Home":
    st.markdown("<h3>Welcome to the Diamond Price Prediction App!</h3>", unsafe_allow_html=True)
    st.markdown("""
    This app predicts the price of diamonds based on various features like carat weight, cut, color, and clarity.
    Enter the details of the diamond below and click on the 'Predict' button to get the predicted price.
    """)

    st.markdown("Choose the 'Prediction' option from the sidebar to make predictions.")
elif selected_page == "Prediction":
    st.sidebar.success("Prediction Page")

    carat = st.number_input('Enter the carat')
    cut = st.selectbox('Select quality of the cut', ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
    color = st.selectbox('Select your diamond color', ['D', 'E', 'F', 'G', 'H', 'I', 'J'])
    clarity = st.selectbox('Select the measurement of the diamond', ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])
    depth = st.number_input('Enter the width of the top of the diamond')
    table = st.number_input('Enter the width of the top of the diamond', step=1, format='%d')
    volume = st.number_input('Enter the vol of the diamond')

    

    if st.button('Predict'):
        model_load_path = "rf.pkl"
        with open(model_load_path, 'rb') as file:
            unpickled_model = pickle.load(file)
            
        if color == 'D':
            color = 0
        elif color == 'E':
            color = 1
        elif color == 'F':
            color = 2
        elif color == 'G':
            color = 3
        elif color == 'H':
            color = 4
        elif color == 'I':
            color = 5
        elif color == 'J':
            color = 6
            
            
        
        
        if clarity == 'I1':
            clarity = 0
        elif clarity == 'SI1':
            clarity = 1
        elif clarity == 'SI2':
            clarity = 2
        elif clarity == 'VS1':
            clarity = 3
        elif clarity == 'VS2':
            clarity = 4
        elif clarity == 'VVS1':
            clarity = 5
        elif clarity == 'VVS2':
            clarity = 6  
        elif clarity == 'F1':
            clarity = 7 

        if cut == 'Fair':
            cut = 0
        elif cut == 'Good':
            cut = 1
        elif cut == 'Very Good':
            cut = 2
        elif cut == 'Good':
            cut = 3
        elif cut == 'Premuim':
            cut = 4
        elif cut == 'Ideal':
            cut = 5
        
            
        user_input = pd.DataFrame({
            "carat": [carat],
            "cut": [cut],
            "color": [color],
            "clarity": [clarity],
            "depth": [depth],
            "table": [table],
            "volume": [volume]
        })

        # Preprocess the categorical features
        # encoded_user_input = pd.get_dummies(user_input, columns=["cut", "color", "clarity"])

        # # Align columns with training data
        # X_train_columns = ['carat', 'depth', 'table', 'x', 'y', 'z']
        # encoded_user_input = encoded_user_input.reindex(columns=X_train_columns, fill_value=0)

        # Scale the numerical features
        scaler = StandardScaler()
        scaled_user_input = scaler.fit_transform(user_input)

        # Make predictions using the trained model
        income_prediction = unpickled_model.predict(scaled_user_input)

        # Display the predicted income
        st.write("The predicted price of the diamond is $", np.exp(income_prediction[0]))










