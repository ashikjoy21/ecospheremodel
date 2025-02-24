import os
os.environ["PAFY_BACKEND"] = "internal"  # Set internal backend for pafy

import streamlit as st
import cv2
import pafy
import pickle
import settings
import numpy as np
from PIL import Image
import io
from dotenv import load_dotenv
from debug_utils import log_info, log_error, log_debug
from roboflow_client import RoboflowClient
import requests
import time

load_dotenv()

# Initialize custom Roboflow client
log_info("Initializing Roboflow client")
try:
    api_key = os.environ.get("ROBOFLOW_API_KEY", "")
    if not api_key:
        log_error("ROBOFLOW_API_KEY not found in environment variables")
    
    # Create custom client instead of using Roboflow package
    CLIENT = RoboflowClient(
        api_key=api_key,
        project_id=settings.PROJECT_ID,
        model_version=settings.MODEL_VERSION
    )
    log_info(f"Successfully connected to Roboflow project: {settings.PROJECT_ID}, version: {settings.MODEL_VERSION}")
except Exception as e:
    log_error(f"Failed to initialize Roboflow client: {str(e)}")
    raise

def load_model(model_info=None):
    """
    Returns the Roboflow client for the specified model
    
    Parameters:
        model_info: Dictionary containing project_id and version
    """
    try:
        # Use default model if none specified
        project_id = settings.PROJECT_ID
        version = settings.MODEL_VERSION
        
        # Override with selected model if provided
        if model_info and isinstance(model_info, dict):
            project_id = model_info.get("project_id", project_id)
            version = model_info.get("version", version)
            
        log_info(f"Loading model: {project_id} (version {version})")
        
        # Create client for the specified model
        client = RoboflowClient(
            api_key=os.environ.get("ROBOFLOW_API_KEY", ""),
            project_id=project_id,
            model_version=version
        )
        
        return client
    except Exception as e:
        log_error(f"Failed to load model: {str(e)}")
        raise

def predict_image(image, conf, model_info=None):
    """
    Predict using Roboflow API with enhanced input handling and detailed logging
    
    Parameters:
        image: Image to predict (PIL Image, numpy array, bytes, or file-like object)
        conf: Confidence threshold
        model_info: Dictionary containing project_id and version
    """
    try:
        log_info(f"Starting image prediction with confidence: {conf}")
        log_debug(f"Input image type: {type(image)}")
        
        # Handle direct file uploads from Streamlit
        if hasattr(image, 'read'):
            log_debug("Processing file upload from Streamlit")
            image_bytes = image.read()
            image = Image.open(io.BytesIO(image_bytes))
            log_debug(f"Converted to PIL Image: {image.size}, mode: {image.mode}")
            
        # Convert all other input types appropriately
        if isinstance(image, bytes):
            log_debug(f"Processing bytes input of length: {len(image)}")
            image = Image.open(io.BytesIO(image))
            log_debug(f"Converted bytes to PIL Image: {image.size}, mode: {image.mode}")
        elif isinstance(image, str):
            log_debug(f"Processing image path: {image}")
            image = Image.open(image)
            log_debug(f"Loaded image from path: {image.size}, mode: {image.mode}")
        elif isinstance(image, np.ndarray):
            log_debug(f"Processing numpy array with shape: {image.shape}, dtype: {image.dtype}")
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                log_debug("Converted BGR to RGB")
            image = Image.fromarray(image)
            log_debug(f"Converted numpy array to PIL Image: {image.size}, mode: {image.mode}")
        elif isinstance(image, Image.Image):
            log_debug(f"Already PIL Image: {image.size}, mode: {image.mode}")
        else:
            error_msg = f"Unsupported image type: {type(image)}"
            log_error(error_msg)
            raise ValueError(error_msg)

        # Convert to bytes
        log_debug("Converting image to JPEG bytes")
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        log_debug(f"Converted to bytes, size: {len(img_byte_arr)} bytes")

        # After converting to bytes, use the selected model
        project_id = settings.PROJECT_ID
        version = settings.MODEL_VERSION
        
        # Override with selected model if provided
        if model_info and isinstance(model_info, dict):
            project_id = model_info.get("project_id", project_id)
            version = model_info.get("version", version)
            
        log_info(f"Using model: {project_id} (version {version}) for prediction")
        
        # Use the selected model for prediction
        url = f"https://detect.roboflow.com/{project_id}/{version}"
        params = {
            "api_key": os.environ.get("ROBOFLOW_API_KEY", ""),
            "confidence": conf,
            "format": "json"
        }
        
        files = {
            "file": ("image.jpg", img_byte_arr, "image/jpeg")
        }
        
        try:
            response = requests.post(url, params=params, files=files)
            
            # Handle access forbidden errors
            if response.status_code == 403:
                log_error(f"Access forbidden to model {project_id}/{version}.")
                
                # Check if we're already using the default model
                if project_id != settings.PROJECT_ID or version != settings.MODEL_VERSION:
                    log_info("Falling back to default model.")
                    
                    # Fall back to the default model
                    project_id = settings.PROJECT_ID
                    version = settings.MODEL_VERSION
                    
                    # Try again with the default model
                    url = f"https://detect.roboflow.com/{project_id}/{version}"
                    response = requests.post(url, params=params, files=files)
            
            # Continue with normal error handling
            if response.status_code != 200:
                error_msg = f"API error: {response.status_code} - {response.text}"
                log_error(error_msg)
                raise Exception(error_msg)
                
        except requests.exceptions.RequestException as req_err:
            log_error(f"Request error: {str(req_err)}")
            # Fall back to default model on connection error
            if project_id != settings.PROJECT_ID or version != settings.MODEL_VERSION:
                log_info("Connection error. Falling back to default model.")
                project_id = settings.PROJECT_ID
                version = settings.MODEL_VERSION
                url = f"https://detect.roboflow.com/{project_id}/{version}"
                response = requests.post(url, params=params, files=files)
        
        # Parse the JSON response
        result = response.json()
        log_info(f"Raw API response: {result}")
        
        # Extract the predictions array from the response
        if 'predictions' in result:
            predictions = result['predictions']
            log_info(f"Received prediction with {len(predictions)} objects detected")
            return predictions  # Return just the predictions array
        else:
            log_error(f"Unexpected response format: {result}")
            return []  # Return empty list if no predictions found
    except Exception as e:
        error_msg = f"Prediction error: {str(e)}"
        log_error(error_msg)
        st.error(error_msg)
        return []  # Return empty list on error

def display_tracker_options():
    display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = True if display_tracker == 'Yes' else False
    if is_display_tracker:
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker, tracker_type
    return is_display_tracker, None

def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None, model_info=None):
    """
    Display the detected objects using Roboflow predictions
    """
    try:
        # Resize the image
        image = cv2.resize(image, (720, int(720*(9/16))))
        
        # Convert BGR to RGB for display
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get predictions with the selected model
        predictions = predict_image(image, conf, model_info)
        
        # Draw predictions on image
        for prediction in predictions:
            # Access bbox values - handle different formats
            if 'bbox' in prediction:
                # Format [x, y, width, height]
                x, y, w, h = prediction['bbox']
            elif all(k in prediction for k in ['x', 'y', 'width', 'height']):
                # Format with separate keys (Roboflow API format)
                x = prediction['x']
                y = prediction['y']
                w = prediction['width']
                h = prediction['height']
            else:
                log_error(f"Unknown bbox format in prediction: {prediction}")
                continue
            
            # Get class and confidence
            label = prediction.get('class', prediction.get('label', 'unknown'))
            confidence = prediction.get('confidence', 0)
            
            if confidence > conf:
                cv2.rectangle(image_rgb, 
                            (int(x), int(y)), 
                            (int(x+w), int(y+h)), 
                            (0, 255, 0), 2)
                cv2.putText(image_rgb, 
                          f'{label}: {confidence:.2f}', 
                          (int(x), int(y-10)), 
                          cv2.FONT_HERSHEY_SIMPLEX, 
                          0.5, (0, 255, 0), 2)
        
        # Display the image using PIL
        pil_image = Image.fromarray(image_rgb)
        st_frame.image(pil_image, use_container_width=True)
        
    except Exception as e:
        log_error(f"Error in display frames: {str(e)}")
        st.error(f"Error in display frames: {str(e)}")

def play_youtube_video(conf, model):
    """
    Plays a webcam stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_youtube = st.sidebar.text_input("YouTube Video url")

    is_display_tracker, tracker = display_tracker_options()

    if st.sidebar.button('Detect Trash'):
        try:
            video = pafy.new(source_youtube)
            best = video.getbest(preftype="mp4")
            vid_cap = cv2.VideoCapture(best.url)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))


#def play_rtsp_stream(conf, model):
    """
    Plays an rtsp stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_rtsp = st.sidebar.text_input("rtsp stream url")
    is_display_tracker, tracker = display_tracker_options()
    if st.sidebar.button('Detect trash'):
        try:
            vid_cap = cv2.VideoCapture(source_rtsp)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading RTSP stream: " + str(e))


def play_webcam(conf, model_info=None):
    """
    Plays a webcam stream with live object detection using Roboflow.
    """
    source_webcam = settings.WEBCAM_PATH
    is_display_tracker, tracker = display_tracker_options()
    
    # Create placeholder for webcam feed
    st_frame = st.empty()
    
    # Create a button to start/stop the webcam
    start_button = st.sidebar.button('Start Webcam Detection')
    stop_button_placeholder = st.sidebar.empty()
    
    # Show status indicator
    status_placeholder = st.sidebar.empty()
    preview_placeholder = st.empty()
    
    # Show instructions
    st.sidebar.info("👆 Click 'Start Webcam Detection' to begin real-time trash detection")
    
    # Add information about detected objects
    detection_info = st.sidebar.empty()
    
    if start_button:
        try:
            # Display initializing message
            status_placeholder.info("Initializing webcam...")
            
            # Create a stop button
            stop_button = stop_button_placeholder.button('Stop Detection')
            
            # Initialize the webcam
            vid_cap = cv2.VideoCapture(source_webcam)
            if not vid_cap.isOpened():
                status_placeholder.error("❌ Could not open webcam. Please check your camera connection.")
                return
                
            status_placeholder.success("✅ Webcam is now active - detecting trash in real-time")
            
            # Loop to continuously get frames from webcam
            detection_count = {}
            frame_count = 0
            
            while vid_cap.isOpened():
                # Check if stop button was clicked
                if stop_button:
                    status_placeholder.info("Stopping webcam...")
                    break
                    
                # Read frame from webcam
                success, image = vid_cap.read()
                if success:
                    try:
                        # Increment frame count
                        frame_count += 1
                        
                        # Process and display the frame with detections
                        predictions = predict_image(image, conf, model_info)
                        
                        # Update detection stats
                        for prediction in predictions:
                            if prediction.get('confidence', 0) > conf:
                                obj_class = prediction.get('class', 'unknown')
                                if obj_class in detection_count:
                                    detection_count[obj_class] += 1
                                else:
                                    detection_count[obj_class] = 1
                        
                        # Display detection stats every 10 frames
                        if frame_count % 10 == 0:
                            info_text = "### Detected Objects\n"
                            if detection_count:
                                for obj_class, count in detection_count.items():
                                    info_text += f"- {obj_class}: {count} instances\n"
                            else:
                                info_text += "No objects detected yet"
                            detection_info.markdown(info_text)
                            
                        # Display frame with detection boxes
                        _display_detected_frames(conf, None, st_frame, image, is_display_tracker, tracker, model_info)
                        
                    except Exception as e:
                        log_error(f"Frame processing error: {str(e)}")
                        continue
                else:
                    log_error("Failed to read frame from webcam")
                    break
                    
                # Add a small delay to control frame rate
                time.sleep(0.01)
                
            # Release resources
            vid_cap.release()
            status_placeholder.info("Webcam stopped")
            
        except Exception as e:
            error_msg = f"Error in webcam stream: {str(e)}"
            log_error(error_msg)
            status_placeholder.error(f"❌ {error_msg}")
        
        finally:
            # Reset button state
            stop_button_placeholder.empty()
            # Display message when done
            st.sidebar.info("👆 Click 'Start Webcam Detection' to begin again")


def play_stored_video(conf, model):
    """
    Plays a stored video file. Tracks and detects objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_vid = st.sidebar.selectbox(
        "Choose a video...", settings.VIDEOS_DICT.keys())

    is_display_tracker, tracker = display_tracker_options()

    with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)

    if st.sidebar.button('Detect Video Trash'):
        try:
            vid_cap = cv2.VideoCapture(
                str(settings.VIDEOS_DICT.get(source_vid)))
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))
