import cv2
import numpy as np
from typing import Tuple, List, Optional


class ImageUtils:
    @staticmethod
    def get_coordinates(width: int, height: int, bbox: np.ndarray) -> List[int]:
        """
        Get face bounding box coordinates with boundary checks

        Args:
            width: Frame width
            height: Frame height
            bbox: Bounding box array [xmin, ymin, xmax, ymax]

        Returns:
            List of coordinates [xmin, ymin, xmax, ymax]
        """
        x1, y1, x2, y2 = bbox.astype(int)
        xmin, ymin = max(0, x1), max(0, y1)
        xmax, ymax = min(width, x2), min(height, y2)
        return [xmin, ymin, xmax, ymax]

    @staticmethod
    def draw_bounding_box(
        frame: np.ndarray, box: List[int], id_name: str, name: Optional[str], sim: float
    ) -> None:
        """
        Draw bounding box around face with identification text

        Args:
            frame: Input frame to draw on
            box: Bounding box coordinates [xmin, ymin, xmax, ymax]
            id_name: User ID or "Unknown"
            name: User name (optional)
            sim: Similarity score
        """
        xmin, ymin, xmax, ymax = box

        # Choose color: green for known, red for unknown
        color = (0, 255, 0) if id_name != "Unknown" else (0, 0, 255)

        # Draw bounding box
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

        # Add text for recognized faces
        if id_name != "Unknown" and name:
            text = f"{name} ({sim:.2f})"
            cv2.putText(
                frame,
                text,
                (xmin, ymin - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )

    @staticmethod
    def get_face_center(box: List[int]) -> Tuple[int, int]:
        """
        Calculate center of face bounding box

        Args:
            box: Bounding box coordinates [xmin, ymin, xmax, ymax]

        Returns:
            Tuple of (x_center, y_center)
        """
        xmin, ymin, xmax, ymax = box
        x_center = int((xmax + xmin) // 2)
        y_center = int((ymax + ymin) // 2)
        return (x_center, y_center)

    @staticmethod
    def add_padding(
        frame: np.ndarray, width: int, height: int, box: List[int], padding: int = 30
    ) -> np.ndarray:
        """
        Add extra padding to face region for better cropping

        Args:
            frame: Input frame
            width: Frame width
            height: Frame height
            box: Bounding box coordinates [xmin, ymin, xmax, ymax]
            padding: Padding size in pixels

        Returns:
            Padded face image
        """
        x1, y1, x2, y2 = box
        xmin = max(0, x1 - padding)
        ymin = max(0, y1 - padding)
        xmax = min(width, x2 + padding)
        ymax = min(height, y2 + padding)

        return frame[ymin:ymax, xmin:xmax]

    @staticmethod
    def resize_frame(frame: np.ndarray, scale: float = 0.5) -> np.ndarray:
        """
        Resize frame for display or processing

        Args:
            frame: Input frame
            scale: Scaling factor

        Returns:
            Resized frame
        """
        if scale == 1.0:
            return frame

        new_width = int(frame.shape[1] * scale)
        new_height = int(frame.shape[0] * scale)
        return cv2.resize(frame, (new_width, new_height))

    @staticmethod
    def preprocess_frame(
        frame: np.ndarray, target_size: Tuple[int, int] = (640, 640)
    ) -> np.ndarray:
        """
        Preprocess frame for face detection

        Args:
            frame: Input frame
            target_size: Target size for processing

        Returns:
            Preprocessed frame
        """
        # Convert to RGB if needed
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame

        # Resize if needed
        if frame_rgb.shape[:2] != target_size:
            frame_rgb = cv2.resize(frame_rgb, target_size)

        return frame_rgb

    @staticmethod
    def draw_regions(frame: np.ndarray, block_points: List, seat_points: List) -> None:
        """
        Draw block and seat regions on frame for visualization

        Args:
            frame: Input frame to draw on
            block_points: List of block regions
            seat_points: List of seat regions
        """
        # Draw block regions
        for block_no, points in block_points:
            cv2.polylines(
                frame, [points], isClosed=True, color=(0, 255, 0), thickness=2
            )
            # Add block number text
            centroid = np.mean(points, axis=0)[0].astype(int)
            cv2.putText(
                frame,
                f"Block {block_no}",
                (centroid[0], centroid[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        # Draw seat regions
        for seat_no, x_min, y_min, x_max, y_max in seat_points:
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            cv2.putText(
                frame,
                f"Seat {seat_no}",
                (x_min, y_min - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )

    @staticmethod
    def enhance_image_quality(image: np.ndarray) -> np.ndarray:
        """
        Enhance image quality for better face recognition

        Args:
            image: Input image

        Returns:
            Enhanced image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        # Merge channels and convert back to BGR
        enhanced_lab = cv2.merge([l, a, b])
        enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        return enhanced_image

    @staticmethod
    def create_montage(
        images: List[np.ndarray], grid_size: Tuple[int, int] = (4, 4)
    ) -> np.ndarray:
        """
        Create a montage of multiple face images

        Args:
            images: List of face images
            grid_size: Grid size for montage (rows, cols)

        Returns:
            Montage image
        """
        if not images:
            return np.zeros((100, 100, 3), dtype=np.uint8)

        rows, cols = grid_size
        montage_height = rows * 100
        montage_width = cols * 100
        montage = np.zeros((montage_height, montage_width, 3), dtype=np.uint8)

        for idx, image in enumerate(images):
            if idx >= rows * cols:
                break

            # Resize image
            resized = cv2.resize(image, (100, 100))

            # Calculate position
            row = idx // cols
            col = idx % cols
            y1, y2 = row * 100, (row + 1) * 100
            x1, x2 = col * 100, (col + 1) * 100

            montage[y1:y2, x1:x2] = resized

        return montage
