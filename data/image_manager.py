import os
import cv2
from datetime import datetime
from config.settings import TRACK_IMAGES_DIR
from utils.image_utils import ImageUtils


class ImageManager(ImageUtils):
    def __init__(self):
        super().__init__()
        self.current_date = datetime.now().strftime("%Y-%m-%d")
        self.track_images_path = TRACK_IMAGES_DIR / self.current_date
        self.track_images_path.mkdir(exist_ok=True)

    def save_track_face(self, face_image, id_name, name, sim):
        """Save tracked face images"""
        folder_path = self.track_images_path / f"{name}_{id_name}"
        folder_path.mkdir(exist_ok=True)

        track_image_time = datetime.now().strftime("%H_%M_%S")
        filename = f"{name}_{track_image_time}_{sim:.2f}.jpg"
        file_path = folder_path / filename

        cv2.imwrite(str(file_path), face_image)
