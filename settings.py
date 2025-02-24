from pathlib import Path
import sys

# Get the absolute path of the current file
file_path = Path(__file__).resolve()

# Get the parent directory of the current file
root_path = file_path.parent

# Add the root path to the sys.path list if it is not already there
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Get the relative path of the root directory with respect to the current working directory
ROOT = root_path.relative_to(Path.cwd())

# Sources
IMAGE = 'Image'
#VIDEO = 'Video'
WEBCAM = 'Webcam'
#RTSP = 'RTSP'
#YOUTUBE = 'YouTube'

SOURCES_LIST = [IMAGE, WEBCAM]

# Images config
IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / 'def.jfif'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'def1.jpg'


# Videos config
VIDEO_DIR = ROOT / 'videos'
VIDEO_1_PATH = VIDEO_DIR / 'video_1.mp4'
VIDEO_2_PATH = VIDEO_DIR / 'video_2.mp4'
VIDEO_3_PATH = VIDEO_DIR / 'video_3.mp4'
VIDEO_4_PATH = VIDEO_DIR / 'video_4.mp4'
VIDEO_5_PATH = VIDEO_DIR / 'video_5.mp4'
VIDEOS_DICT = {
    'video_1': VIDEO_1_PATH,
    'video_2': VIDEO_2_PATH,
    'video_3': VIDEO_3_PATH,
    'video_4': VIDEO_4_PATH,
    'video_5': VIDEO_5_PATH,
}


# ML Model config
MODEL_DIR = None  # Not needed for Roboflow API
DETECTION_MODEL = None  # Not needed for Roboflow API
#SEGMENTATION_MODEL = MODEL_DIR+'yolov8n-seg.pt'

# Webcam
WEBCAM_PATH = 0

# Roboflow config
USE_ROBOFLOW = True
MODEL_ID = "trash_detect-subnn/3"

# Parse MODEL_ID to get PROJECT_ID and MODEL_VERSION
MODEL_ID_PARTS = MODEL_ID.split("/")
PROJECT_ID = MODEL_ID_PARTS[0]  # "trash_detect-subnn"
MODEL_VERSION = MODEL_ID_PARTS[1]  # "3"

# Available Roboflow models
ROBOFLOW_MODELS = {
    "Trash Detection": {
        "project_id": "trash_detect-subnn",
        "version": "3",
        "description": "Detects and classifies various types of waste materials",
        "public": True
    },
    "Garbage Classification": {
        "project_id": "garbage-classification-3",
        "version": "2",
        "description": "Advanced classification of recyclable and non-recyclable waste",
        "public": True
    }
}

# Default model
DEFAULT_MODEL = "Trash Detection"
