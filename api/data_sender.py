import time
import requests
from data.json_manager import JSONManager
from config.settings import (
    TRACK_TEMP_FILE,
    TRACKING_DATA_FILE,
    CHECK_TRACKING_DATA_FILE,
)
from config.constants import (
    API_URL,
    HEADERS,
    API_RETRY_COUNT,
    API_TIMEOUT,
    API_BATCH_SIZE,
)


class DataSender:
    def __init__(self):
        self.api_url = API_URL
        self.headers = HEADERS
        self.retry_count = API_RETRY_COUNT
        self.timeout = API_TIMEOUT
        self.batch_size = API_BATCH_SIZE

    def send_track_data(self, stop_event):
        """Send track data to API periodically with retry logic"""
        while not stop_event.is_set():
            time.sleep(10)
            try:
                tracking_data = JSONManager.safe_load_json(TRACK_TEMP_FILE, default=[])

                if not tracking_data:
                    continue

                # Clear temporary file
                JSONManager.safe_write_json(TRACK_TEMP_FILE, [])

                # Verify data was cleared
                tracking_data_check = JSONManager.safe_load_json(
                    TRACK_TEMP_FILE, default=[]
                )
                if tracking_data_check:
                    print(
                        f"Verification Failed: data.json is NOT empty! Data: {tracking_data_check}"
                    )
                    continue

                print(
                    f"Verification Passed: data.json is empty! Sending {len(tracking_data)} records."
                )

                # Update tracking files
                all_data = JSONManager.safe_load_json(TRACKING_DATA_FILE, default=[])
                check_all_data = JSONManager.safe_load_json(
                    CHECK_TRACKING_DATA_FILE, default=[]
                )

                all_data.extend(tracking_data)
                check_all_data.extend(tracking_data)

                JSONManager.safe_write_json(TRACKING_DATA_FILE, all_data)
                JSONManager.safe_write_json(CHECK_TRACKING_DATA_FILE, check_all_data)

                # Send to API with retry logic
                success = self._send_with_retry(all_data)

                if success:
                    print("Track Data sent successfully.")
                    JSONManager.safe_write_json(TRACKING_DATA_FILE, [])
                else:
                    print("Failed to send track data after retries.")

            except Exception as e:
                print(f"Error in send_track_data: {e}")

    def _send_with_retry(self, data):
        """Send data with retry logic"""
        for attempt in range(self.retry_count):
            try:
                response = requests.post(
                    self.api_url, json=data, headers=self.headers, timeout=self.timeout
                )

                if response.status_code == 200:
                    return True
                else:
                    print(
                        f"Attempt {attempt + 1} failed: Status {response.status_code}, Response: {response.text}"
                    )

            except requests.exceptions.RequestException as e:
                print(f"Attempt {attempt + 1} failed with exception: {e}")

            if attempt < self.retry_count - 1:
                time.sleep(2**attempt)  # Exponential backoff

        return False
