import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
API_URL = os.getenv("API_URL", "https://your-default-api.com/api/track")
API_KEY = os.getenv("API_KEY", "default_api_key")
HEADERS = {"Content-Type": "application/json", "ApiKey": API_KEY}

# Camera Configuration
CAMERA_RTSP_USERNAME = os.getenv("CAMERA_RTSP_USERNAME", "admin")
CAMERA_RTSP_PASSWORD = os.getenv("CAMERA_RTSP_PASSWORD", "password")

# Model Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "hybrid2")
MODEL_PROVIDERS = os.getenv(
    "MODEL_PROVIDERS", "CUDAExecutionProvider,CPUExecutionProvider"
).split(",")

# Processing Configuration
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))
DETECTION_THRESHOLD = float(os.getenv("DETECTION_THRESHOLD", "0.33"))
SKIP_FRAMES_WORKING = int(os.getenv("SKIP_FRAMES_WORKING", "5"))
SKIP_FRAMES_IDLE = int(os.getenv("SKIP_FRAMES_IDLE", "5"))
RECOGNITION_COOLDOWN = int(os.getenv("RECOGNITION_COOLDOWN", "60"))

# System Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
SAVE_TRACK_IMAGES = os.getenv("SAVE_TRACK_IMAGES", "true").lower() == "true"
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
MAX_PROCESSES = int(os.getenv("MAX_PROCESSES", "8"))

# API Configuration
API_RETRY_COUNT = int(os.getenv("API_RETRY_COUNT", "3"))
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "10"))
API_BATCH_SIZE = int(os.getenv("API_BATCH_SIZE", "50"))

# Region Detection
ENABLE_BLOCK_REGIONS = os.getenv("ENABLE_BLOCK_REGIONS", "true").lower() == "true"
ENABLE_SEAT_REGIONS = os.getenv("ENABLE_SEAT_REGIONS", "true").lower() == "true"

# Performance Tuning
USE_GPU = os.getenv("USE_GPU", "true").lower() == "true"
OPTIMIZE_MEMORY = os.getenv("OPTIMIZE_MEMORY", "true").lower() == "true"
ENABLE_FRAME_SKIPPING = os.getenv("ENABLE_FRAME_SKIPPING", "true").lower() == "true"

# Camera URLs (now with environment variable support)
CAMERA_URLS = [
    f"rtsp://172.14.0.187:554/rtsp/streaming?channel=01&subtype=0&transport=tcp",
    f"rtsp://{CAMERA_RTSP_USERNAME}:{CAMERA_RTSP_PASSWORD}@172.14.0.112:554/?transport=tcp",
    f"rtsp://{CAMERA_RTSP_USERNAME}:{CAMERA_RTSP_PASSWORD}@172.14.0.116:554/?transport=tcp",
    f"rtsp://{CAMERA_RTSP_USERNAME}:{CAMERA_RTSP_PASSWORD}@172.14.0.117:554/?transport=tcp",
    f"rtsp://{CAMERA_RTSP_USERNAME}:{CAMERA_RTSP_PASSWORD}@172.14.0.119:554/?transport=tcp",
    f"rtsp://{CAMERA_RTSP_USERNAME}:{CAMERA_RTSP_PASSWORD}@172.14.0.120:554/?transport=tcp",
    f"rtsp://{CAMERA_RTSP_USERNAME}:{CAMERA_RTSP_PASSWORD}@172.14.0.121:554/?transport=tcp",
    f"rtsp://{CAMERA_RTSP_USERNAME}:{CAMERA_RTSP_PASSWORD}@172.14.0.122:554/?transport=tcp",
    f"rtsp://{CAMERA_RTSP_USERNAME}:{CAMERA_RTSP_PASSWORD}@172.14.0.124:554/?transport=tcp",
    f"rtsp://{CAMERA_RTSP_USERNAME}:{CAMERA_RTSP_PASSWORD}@172.14.0.181:554/?transport=tcp",
    f"rtsp://{CAMERA_RTSP_USERNAME}:{CAMERA_RTSP_PASSWORD}@172.14.0.182:554/?transport=tcp",
    f"rtsp://{CAMERA_RTSP_USERNAME}:{CAMERA_RTSP_PASSWORD}@172.14.0.183:554/?transport=tcp",
]
