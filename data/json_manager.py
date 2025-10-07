import json
import os


class JSONManager:
    @staticmethod
    def safe_load_json(file_path, default=None):
        """Safely loads JSON data from a file"""
        try:
            if os.path.exists(file_path) and os.stat(file_path).st_size > 0:
                with open(file_path, "r") as file:
                    return json.load(file)
        except (json.JSONDecodeError, FileNotFoundError, OSError):
            pass
        return default if default is not None else []

    @staticmethod
    def safe_write_json(file_path, data):
        """Writes JSON data safely to a file"""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w") as file:
                json.dump(data, file, indent=4)
        except Exception as e:
            print(f"Error writing to JSON file {file_path}: {e}")
