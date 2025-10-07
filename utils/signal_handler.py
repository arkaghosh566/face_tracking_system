import signal
from config.settings import TRACK_TEMP_FILE, TRACKING_DATA_FILE
from data.json_manager import JSONManager


class SignalHandler:
    def __init__(self, stop_event):
        self.stop_event = stop_event

    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print("Signal received, stopping processes...")
        self.stop_event.set()

    @staticmethod
    def clear_cache_and_data():
        """Clear track data files"""
        try:
            for file_path in [TRACK_TEMP_FILE, TRACKING_DATA_FILE]:
                if file_path.exists():
                    JSONManager.safe_write_json(file_path, [])
            print("Data cleared.")
        except Exception as e:
            print(f"Error clearing cache and data: {e}")
