import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

# Path configurations
TRACK_DATA_DIR = BASE_DIR / "track_data"
TRACK_RECORDS_DIR = BASE_DIR / "track_records"
TRACK_IMAGES_DIR = BASE_DIR / "track_images"
MODELS_DIR = BASE_DIR / "models" / "hybrid2"

# Create directories
TRACK_DATA_DIR.mkdir(exist_ok=True)
TRACK_RECORDS_DIR.mkdir(exist_ok=True)
TRACK_IMAGES_DIR.mkdir(exist_ok=True)
MODELS_DIR.parent.mkdir(exist_ok=True)

# File paths
TRACK_TEMP_FILE = TRACK_DATA_DIR / "data.json"
CHECK_TRACK_TEMP_FILE = TRACK_DATA_DIR / "check_data.json"
TRACKING_DATA_FILE = TRACK_DATA_DIR / "track.json"
CHECK_TRACKING_DATA_FILE = TRACK_DATA_DIR / "check_track.json"

# JSON configuration files
FEATURES_FILE = BASE_DIR / "Office2_Hybrid2_face1_feat.json"
NAMES_FILE = BASE_DIR / "json_folder" / "office2.json"
BLOCK_REGIONS_FILE = BASE_DIR / "json_folder" / "new_office_block.json"
SEAT_REGIONS_FILE = BASE_DIR / "json_folder" / "new_office_seat.json"
CAMERA_FILE = BASE_DIR / "json_folder" / "new_camera_ip.json"
