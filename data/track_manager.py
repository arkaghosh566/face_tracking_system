import os
from datetime import datetime
from data.json_manager import JSONManager
from config.settings import TRACK_TEMP_FILE, CHECK_TRACK_TEMP_FILE, TRACK_RECORDS_DIR


class TrackManager:
    def __init__(self):
        self.current_date = datetime.now().strftime("%Y-%m-%d")
        self.track_records_path = TRACK_RECORDS_DIR / self.current_date
        self.track_records_path.mkdir(exist_ok=True)

    def mark_track_data(self, employee_id, camera_ip, block_no, seat_no):
        """Mark track data for recognized individuals"""
        try:
            employee_file_path = self.track_records_path / f"{employee_id}.json"
            current_time = datetime.now().strftime("%H:%M:%S")

            # Create new entry
            new_entry = {
                "userPin": employee_id,
                "date": self.current_date,
                "time": current_time,
                "camIP": camera_ip,
                "region": block_no,
                "seat": seat_no,
            }

            # Update employee-specific file
            employee_data = JSONManager.safe_load_json(employee_file_path, default=[])
            employee_data.append(new_entry)
            JSONManager.safe_write_json(employee_file_path, employee_data)

            # Update temporary track files
            track_data = JSONManager.safe_load_json(TRACK_TEMP_FILE, default=[])
            check_track_data = JSONManager.safe_load_json(
                CHECK_TRACK_TEMP_FILE, default=[]
            )

            track_data.append(new_entry)
            check_track_data.append(new_entry)

            JSONManager.safe_write_json(TRACK_TEMP_FILE, track_data)
            JSONManager.safe_write_json(CHECK_TRACK_TEMP_FILE, check_track_data)

        except Exception as e:
            print(f"Error in mark_track_data: {e}")
