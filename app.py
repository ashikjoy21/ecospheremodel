# Python In-built packages
from pathlib import Path
import PIL
import cv2
import numpy as np

# External packages
import streamlit as st

# Local Modules
import settings
import helper
from debug_utils import show_debug_panel, log_info, log_debug, log_error

# Setting page layout with new branding
st.set_page_config(
    page_title="Ecosphere Dustbin Waste Classifier",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading and description
st.title("♻️ Ecosphere Dustbin Waste Classifier")
st.markdown("""
<style>
    .main-header {
        font-family: 'Helvetica', sans-serif;
        color: #ffffff;
    }
    .subheader {
        color: #cccccc;
        font-size: 1.1em;
    }
    .sidebar-header {
        color: #00ff95;
        font-weight: bold;
        font-size: 1.2em;
    }
    .stButton>button {
        background-color: #00ff95;
        color: #000000;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #00cc7a;
    }
    /* Style for model description box */
    .model-desc-box {
        background-color: #2c3e50;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #00ff95;
        color: #ffffff;
    }
    /* Style for the footer */
    .footer {
        text-align: center;
        color: #cccccc;
        padding: 20px 0;
        border-top: 1px solid #00ff95;
        margin-top: 50px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="subheader">
    An intelligent waste classification system that helps identify and categorize different types of waste 
    for better recycling and waste management.
</div>
""", unsafe_allow_html=True)

# Sidebar with new styling
st.sidebar.markdown('<p class="sidebar-header">Model Configuration</p>', unsafe_allow_html=True)

# Model confidence slider with improved styling
confidence = float(st.sidebar.slider(
    "Detection Confidence Threshold", 
    min_value=25, 
    max_value=100, 
    value=40,
    help="Adjust this value to control detection sensitivity"
)) / 100

# Model selection with new styling
st.sidebar.markdown("---")
st.sidebar.markdown('<p class="sidebar-header">Model Selection</p>', unsafe_allow_html=True)

model_options = list(settings.ROBOFLOW_MODELS.keys())
selected_model_name = st.sidebar.selectbox(
    "Select Classification Model", 
    model_options,
    index=model_options.index(settings.DEFAULT_MODEL)
)

# Model description with enhanced styling
selected_model = settings.ROBOFLOW_MODELS[selected_model_name]
st.sidebar.markdown(f"""
<div class="model-desc-box">
    <strong style="color: #00ff95;">Model Description:</strong><br>
    {selected_model['description']}
</div>
""", unsafe_allow_html=True)

# Input source selection
st.sidebar.markdown("---")
st.sidebar.markdown('<p class="sidebar-header">Input Configuration</p>', unsafe_allow_html=True)
source_radio = st.sidebar.radio(
    "Select Input Source",
    settings.SOURCES_LIST,
    help="Choose the source for waste classification"
)

# Pass the selected model to the helper functions
model_info = {
    "name": selected_model_name,
    "project_id": selected_model["project_id"],
    "version": selected_model["version"]
}

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_info)
except Exception as ex:
    st.error("Roboflow connection error:")
    st.error(ex)

source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image",
                         use_container_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_container_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',
                     use_container_width=True)
        else:
            if st.sidebar.button('Detect Objects'):
                try:
                    predictions = helper.predict_image(uploaded_image, confidence, model_info)
                    log_info(f"Predictions type: {type(predictions)}, content: {predictions}")
                    
                    # Convert PIL image to numpy array for drawing
                    image_array = np.array(uploaded_image)
                    
                    # Convert to RGB if needed
                    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                    
                    # Draw predictions
                    if predictions and isinstance(predictions, list):
                        for prediction in predictions:
                            # Make sure prediction is a dictionary
                            if isinstance(prediction, dict):
                                # Access bbox values - some APIs use different formats
                                if 'bbox' in prediction:
                                    # Format [x, y, width, height]
                                    bbox = prediction['bbox']
                                    x, y, w, h = bbox
                                elif all(k in prediction for k in ['x', 'y', 'width', 'height']):
                                    # Format with separate keys
                                    x = prediction['x']
                                    y = prediction['y']
                                    w = prediction['width']
                                    h = prediction['height']
                                else:
                                    log_error(f"Unknown bbox format in prediction: {prediction}")
                                    continue
                                
                                # Get class and confidence
                                label = prediction.get('class', prediction.get('label', 'unknown'))
                                pred_conf = prediction.get('confidence', 0)
                                
                                if pred_conf > confidence:
                                    cv2.rectangle(image_array, 
                                                (int(x), int(y)), 
                                                (int(x+w), int(y+h)), 
                                                (0, 255, 0), 2)
                                    cv2.putText(image_array, 
                                              f'{label}: {pred_conf:.2f}', 
                                              (int(x), int(y-10)), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 
                                              0.5, (0, 255, 0), 2)
                    
                    # Convert back to PIL for display
                    pil_image = PIL.Image.fromarray(image_array)
                    st.image(pil_image, caption='Detected Image', use_container_width=True)
                    
                    with st.expander("Detection Results"):
                        st.write(predictions)
                except Exception as ex:
                    st.error(f"Error during detection: {str(ex)}")
                    log_error(f"Detection exception: {str(ex)}")

elif source_radio == settings.WEBCAM:
    helper.play_webcam(confidence, model_info)

else:
    st.error("Please select a valid source type!")

with st.sidebar:
    st.markdown("---")
    show_debug_panel()

# Add environmental impact section
if st.sidebar.checkbox("Show Environmental Impact", value=False):
    st.sidebar.markdown("""
    <div style="background-color: #2c3e50; padding: 15px; border-radius: 5px; border: 1px solid #00ff95;">
    <h3 style="color: #00ff95;">🌍 Environmental Impact</h3>
    
    <p style="color: #ffffff;">Proper waste classification helps:</p>
    <ul style="color: #ffffff;">
        <li>♻️ Increase recycling efficiency</li>
        <li>🌱 Reduce landfill waste</li>
        <li>💚 Lower carbon footprint</li>
        <li>🌿 Conserve natural resources</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# Update the footer with new styling
st.markdown("""
<div class="footer">
    Powered by Ecosphere AI • Making waste management smarter and sustainable
</div>
""", unsafe_allow_html=True)


