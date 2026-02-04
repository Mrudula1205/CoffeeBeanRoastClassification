import streamlit as st
from PIL import Image
import numpy as np
from src.coffee_roast_ai.preprocessing import process_image
from src.coffee_roast_ai.model_engine import CoffeeModelEngine
from src.coffee_roast_ai.utils import read_params

# 1. Page Configuration & UI
st.set_page_config(page_title="Coffee Roast Quality AI", layout="centered")
st.title("☕ Coffee Roast Quality Control")
st.subheader("Validate roast consistency against production standards")

# 2. Initialization (Using our modular engine)
@st.cache_resource
def initialize_engine():
    config = read_params()
    engine = CoffeeModelEngine()
    # Assuming your best model is saved here
    engine.load_existing_model("models/inception_v1.hdf5") 
    return engine, config

engine, config = initialize_engine()

st.sidebar.header("Quality Standard")
target_roast = st.sidebar.selectbox(
    "What is the target roast for this batch?",
    config['data']['class_names']
)

# 3. File Uploader
file = st.file_uploader("Upload a photo of the roast batch", type=["jpg", "png", "jpeg"])

if file:
    # Display Image
    image = Image.open(file)
    st.image(image, caption=f"Validating against {target_roast} standard", use_container_width=True)
    with st.spinner("Analyzing quality..."):
        # 4. Process & Predict using our modules
        processed_img = process_image(image)
        predictions = engine.model.predict(processed_img)
        
        # 5. Extract Results
        class_names = config['data']['class_names']
        result_idx = np.argmax(predictions)
        confidence = np.max(predictions)
        result_label = class_names[result_idx]

        # 6. Quality Validation Logic (The "Redefined" part)
        st.divider()
        st.subheader("QC Verification")
    
    if result_label == target_roast:
        st.success(f"✅ **PASS**: Batch matches the {target_roast} profile.")
        st.metric("Match Confidence", f"{confidence:.2%}")
    else:
        st.error(f"❌ **FAIL**: Roast Mismatch Detected.")
        st.write(f"**Target:** {target_roast}")
        st.write(f"**Actual Detected:** {result_label}")
        st.warning("The beans appear lighter/darker than the target profile. Adjust roaster settings.")