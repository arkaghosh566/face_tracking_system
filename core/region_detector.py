import cv2
import numpy as np
from typing import List, Optional, Tuple
from utils.geometry_utils import (
    get_face_center,
    is_point_in_polygon,
    calculate_polygon_center,
)


class RegionDetector:
    def __init__(self):
        self.block_points = None
        self.seat_points = None
        self.last_frame_width = 0
        self.last_frame_height = 0

    def update_regions(
        self,
        block_regions: Optional[List],
        seat_regions: Optional[List],
        width: int,
        height: int,
    ) -> None:
        """Update region points when frame size changes"""
        if width != self.last_frame_width or height != self.last_frame_height:
            if block_regions:
                self.block_points = self._convert_block_regions(
                    block_regions, width, height
                )
            if seat_regions:
                self.seat_points = self._convert_seat_regions(
                    seat_regions, width, height
                )

            self.last_frame_width = width
            self.last_frame_height = height

    def _convert_block_regions(
        self, block_regions: List, width: int, height: int
    ) -> List:
        """Convert block regions to image coordinates"""
        block_points = []
        for block in block_regions:
            block_no = next(iter(block))
            coordinates = block[block_no]
            scaled_coordinates = [
                [int(x * width), int(y * height)] for x, y in coordinates
            ]
            points = np.array(scaled_coordinates, dtype=np.int32)
            points = points.reshape((-1, 1, 2))
            block_points.append([block_no, points])
        return block_points

    def _convert_seat_regions(
        self, seat_regions: List, width: int, height: int
    ) -> List:
        """Convert seat regions to image coordinates"""
        seat_points = []
        for seat in seat_regions:
            seat_no = next(iter(seat))
            seat_box = seat[seat_no]
            x_min = int(seat_box[0] * width)
            y_min = int(seat_box[1] * height)
            x_max = int((seat_box[0] + seat_box[2]) * width)
            y_max = int((seat_box[1] + seat_box[3]) * height)
            seat_points.append([seat_no, x_min, y_min, x_max, y_max])
        return seat_points

    def check_block_region(
        self, frame: np.ndarray, face_center: Tuple[int, int]
    ) -> Optional[str]:
        """Check if face center is inside any block region"""
        if not self.block_points:
            return None

        for block_no, points in self.block_points:
            cv2.polylines(
                frame, [points], isClosed=True, color=(0, 255, 0), thickness=2
            )
            if is_point_in_polygon(face_center, points):
                return block_no
        return None

    def check_seat_region(
        self, frame: np.ndarray, face_center: Tuple[int, int]
    ) -> Optional[str]:
        """Check if face center is inside any seat region"""
        if not self.seat_points:
            return None

        xcenter, ycenter = face_center
        for seat in self.seat_points:
            seat_no, x_min, y_min, x_max, y_max = seat

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            cv2.putText(
                frame,
                seat_no,
                (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

            if x_min < xcenter < x_max and y_min < ycenter < y_max:
                return seat_no
        return None

    def get_all_regions_visualization(self, frame: np.ndarray) -> np.ndarray:
        """Get frame with all regions visualized"""
        viz_frame = frame.copy()

        if self.block_points:
            for block_no, points in self.block_points:
                cv2.polylines(
                    viz_frame, [points], isClosed=True, color=(0, 255, 0), thickness=2
                )
                center = calculate_polygon_center(points)
                cv2.putText(
                    viz_frame,
                    f"Block {block_no}",
                    center,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

        if self.seat_points:
            for seat_no, x_min, y_min, x_max, y_max in self.seat_points:
                cv2.rectangle(viz_frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                cv2.putText(
                    viz_frame,
                    f"Seat {seat_no}",
                    (x_min, y_min - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2,
                )

        return viz_frame
