
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# Set the page layout to wide
st.set_page_config(layout="wide")

# Load the model once to improve performance
@st.cache
def load_model():
    model = tf.keras.models.load_model('vgg.hdf5')
    return model

# Apply custom styling
st.markdown(
    """
    <style>
    .st-title {
        color: #3E2723;
        font-size: 40px;
        font-weight: bold;
    }
    .st-header {
        color: #6D4C41;
        font-size: 24px;
        font-weight: bold;
    }
    
    .st-success {
        font-size: 18px;
        color: #2e7d32;
        background-color: #a5d6a7;
        padding: 10px;
        border-radius: 5px;
    }
    .st-uploaded-image {
        border: 5px solid #6D4C41;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar for user input selection
st.sidebar.title("Choose Input Source")
input_method = st.sidebar.radio("Select an input method:", ("Upload Image", "Use Camera", "Input Image URL"))

# Title of the app
st.title("Coffee Bean Quality Prediction App")

col1, col2 = st.columns([2, 1])  # Adjust the column ratios as needed

# Column 1: Text
with col1:
   
    st.markdown( """ 
                <div class="col1-box"> 
                <h2>Welcome to the Coffee Bean Quality Prediction App! ☕</h2> 
                <p> Coffee is more than just a drink – it's a passion, an experience, and for many, a daily ritual that starts 
                the day with energy and warmth. Whether you enjoy it bold, smooth, or with a touch of cream, the quality of 
                your coffee beans plays a significant role in how your cup of coffee tastes. This app uses a powerful machine 
                learning model to analyze your coffee beans and predict their quality, helping you get the best brew every time. 
                Factors like color, shape, and size influence the aroma, flavor, and smoothness of your coffee – and we're here to 
                help you determine which beans are ready for your perfect cup. Simply upload an image of your coffee beans, and let 
                the app provide an accurate prediction of their quality. Remember, great coffee starts with great beans! 
                </p> </div> """
                , unsafe_allow_html=True )
  
# Column 2: Image
with col2:
    
    # You can replace the image URL with your local file or a web image URL
    st.image("coffee_beans_image.jpg", caption="Fresh Coffee Beans", use_column_width=True)


# Instructions for the user
st.header("Upload an Image of Coffee Beans to Predict Their Quality")
st.write("This app uses a trained model to classify coffee beans into different quality categories based on an image you upload. Please upload a clear image.")

# Function to preprocess and predict image quality
def import_and_predict(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size)
    img = np.array(image, dtype='float32') / 255.0
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

# Image upload section
if input_method == "Upload Image":
    file = st.file_uploader("Choose a Coffee Bean Image", type=["jpg", "png"])

    if file is not None:
        image = Image.open(file)
        # Display the uploaded image with better styling
        st.image(image, use_column_width=True, caption="Uploaded Image")

        # Prediction logic
        model = load_model()
        predictions = import_and_predict(image, model)
        class_names = ['Dark', 'Green', 'Light', 'Medium']
        predicted_class = class_names[np.argmax(predictions)]
        confidence = round(np.max(predictions), 2)

        # Display the prediction results with improved styling
        result_message = f"This image most likely is: **{predicted_class}** with a confidence of **{confidence*100}%**"
        st.success(result_message)

# Camera input section
elif input_method == "Use Camera":
    # Use camera input feature
    img = st.camera_input("Capture a Coffee Bean Image")

    if img:
        image = Image.open(img)
        # Display the captured image with better styling
        st.image(image, use_column_width=True, caption="Captured Image")

        # Prediction logic
        model = load_model()
        predictions = import_and_predict(image, model)
        class_names = ['Dark', 'Green', 'Light', 'Medium']
        predicted_class = class_names[np.argmax(predictions)]
        confidence = round(np.max(predictions), 2)

        # Display the prediction results with improved styling
        result_message = f"This image most likely is: **{predicted_class}** with a confidence of **{confidence*100}%**"
        st.success(result_message)

# Image URL input section
elif input_method == "Input Image URL":
    image_url = st.text_input("Enter Image URL:")

    if image_url:
        #image = Image.open(image_url)
        # Display the image from URL
        import requests
        from io import BytesIO

        # Fetch the image from the URL
        response = requests.get(image_url)
        response.raise_for_status()  # Check if the request was successful

        # Open the image from the response content
        image = Image.open(BytesIO(response.content))
        st.image(image, use_column_width=True, caption="Image from URL")

        # Prediction logic
        model = load_model()
        predictions = import_and_predict(image, model)
        class_names = ['Dark', 'Green', 'Light', 'Medium']
        predicted_class = class_names[np.argmax(predictions)]
        confidence = round(np.max(predictions), 2)

        # Display the prediction results with improved styling
        result_message = f"This image most likely is: **{predicted_class}** with a confidence of **{confidence*100}%**"
        st.success(result_message)
