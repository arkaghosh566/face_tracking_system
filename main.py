import os
import cupy as cp
from multiprocessing import Process, Event
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import configurations
from config.constants import *
from config.settings import *

# Import modules
from core.face_analyzer import FaceAnalyzer
from core.face_matcher import FaceMatcher
from core.region_detector import RegionDetector
from core.video_processor import VideoProcessor
from data.json_manager import JSONManager
from data.track_manager import TrackManager
from data.image_manager import ImageManager
from api.data_sender import DataSender
from utils.signal_handler import SignalHandler

# Set API configuration from environment
API_URL = os.getenv("API_URL")
HEADERS = {"Content-Type": "application/json", "ApiKey": os.getenv("API_KEY")}


def load_shared_data():
    """Load all shared data for processes"""
    features_list = JSONManager.safe_load_json(FEATURES_FILE, default={})
    names_list = JSONManager.safe_load_json(NAMES_FILE, default={})
    block_regions_list = JSONManager.safe_load_json(BLOCK_REGIONS_FILE, default={})
    seat_regions_list = JSONManager.safe_load_json(SEAT_REGIONS_FILE, default={})
    camera_list = JSONManager.safe_load_json(CAMERA_FILE, default={})

    # Prepare feature matrix
    user_ids = list(features_list.keys())
    feature_matrix = cp.array([cp.array(v).flatten() for v in features_list.values()])
    if feature_matrix.ndim > 2:
        feature_matrix = feature_matrix.reshape(feature_matrix.shape[0], -1)

    return {
        "features": [user_ids, feature_matrix],
        "block_regions": block_regions_list,
        "seat_regions": seat_regions_list,
        "names": names_list,
        "camera_status": camera_list,
    }


def main():
    # Initialize
    stop_event = Event()
    signal_handler = SignalHandler(stop_event)

    # Clear existing data and setup signal handlers
    signal_handler.clear_cache_and_data()
    signal_handler.setup_signal_handlers()

    # Load shared data
    shared_data = load_shared_data()

    # Create and start processes
    processes = []
    for camera_url in CAMERA_URLS:
        processor = VideoProcessor(camera_url, shared_data, stop_event)
        process = Process(
            target=processor.process_stream,
            args=(
                FaceAnalyzer(),
                FaceMatcher(),
                RegionDetector(),
                TrackManager(),
                ImageManager(),
            ),
        )
        processes.append(process)

    # Start data sender
    data_sender = DataSender()
    track_process = Process(target=data_sender.send_track_data, args=(stop_event,))
    track_process.start()

    # Start camera processes
    for process in processes:
        process.start()

    # Wait for processes to complete
    try:
        for process in processes:
            process.join()
        track_process.join()
    except KeyboardInterrupt:
        print("Shutting down...")
        stop_event.set()

    print("System stopped.")


if __name__ == "__main__":
    main()
