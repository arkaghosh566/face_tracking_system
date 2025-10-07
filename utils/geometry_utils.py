import cv2
import numpy as np
from typing import Tuple, List


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


def calculate_iou(box1: List[int], box2: List[int]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes

    Args:
        box1: First bounding box [x1, y1, x2, y2]
        box2: Second bounding box [x1, y1, x2, y2]

    Returns:
        IoU score
    """
    # Calculate intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def is_point_in_polygon(point: Tuple[int, int], polygon: np.ndarray) -> bool:
    """
    Check if a point is inside a polygon

    Args:
        point: (x, y) coordinates to check
        polygon: Numpy array of polygon points

    Returns:
        True if point is inside polygon
    """
    return cv2.pointPolygonTest(polygon, point, measureDist=False) >= 0


def calculate_polygon_center(polygon: np.ndarray) -> Tuple[int, int]:
    """
    Calculate the center point of a polygon

    Args:
        polygon: Numpy array of polygon points

    Returns:
        Center point (x, y)
    """
    moments = cv2.moments(polygon)
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
    else:
        # Fallback: use mean of points
        points = polygon.reshape(-1, 2)
        cx = int(np.mean(points[:, 0]))
        cy = int(np.mean(points[:, 1]))
    return (cx, cy)
