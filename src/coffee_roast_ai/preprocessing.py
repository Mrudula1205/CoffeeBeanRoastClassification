import numpy as np
from PIL import Image


def process_image(image_input, target_size=(224, 224)):
    """
    Standardizes input images (from path or PIL) for the model.
    Ensures consistency between training and inference.
    """
    # 1. Handle different input types (Path string vs PIL Image from Streamlit)
    if isinstance(image_input, str):
        image = Image.open(image_input).convert('RGB')
    else:
        image = image_input.convert('RGB')

    # 2. Resize using PIL (no libGL dependency)
    image = image.resize(target_size)
    image = np.array(image)

    # 3. Rescale (Crucial: matching your 1./255 logic)
    image = image.astype('float32') / 255.0

    # 4. Expand dimensions for the model (Batch size of 1)
    image = np.expand_dims(image, axis=0)

    return image


def get_color_metrics(image_array):
    """
    Experimental: For your 'Golden Dataset' closeness logic.
    Calculates the mean color of the pixels.
    """
    # Remove the batch dimension for calculation
    img = np.squeeze(image_array)
    return np.mean(img, axis=(0, 1))
