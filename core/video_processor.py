import cv2
import re
import time
from datetime import datetime
from config.constants import SKIP_FRAMES_WORKING, SKIP_FRAMES_IDLE, RECOGNITION_COOLDOWN


class VideoProcessor:
    def __init__(self, camera_url, shared_data, stop_event):
        self.camera_url = camera_url
        self.shared_data = shared_data
        self.stop_event = stop_event
        self.ip_address = self._extract_ip_address()
        self.skip_frames = self._get_skip_frames()
        self.last_recognition_times = {}

    def _extract_ip_address(self):
        """Extract IP address from camera URL"""
        ip_match = re.search(r"(\d+\.\d+\.\d+\.\d+)", self.camera_url)
        return ip_match.group(1) if ip_match else None

    def _get_skip_frames(self):
        """Determine number of frames to skip based on camera status"""
        camera_list = self.shared_data["camera_status"]
        if self.ip_address in camera_list:
            return (
                SKIP_FRAMES_WORKING
                if camera_list[self.ip_address] == "Working"
                else SKIP_FRAMES_IDLE
            )
        return SKIP_FRAMES_IDLE

    def process_stream(
        self, face_analyzer, face_matcher, region_detector, track_manager, image_manager
    ):
        """Main video processing loop"""
        while not self.stop_event.is_set():
            capture = cv2.VideoCapture(self.camera_url, cv2.CAP_FFMPEG)
            capture.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)

            if not capture.isOpened():
                print(f"Failed to open {self.camera_url}, retrying in 10 seconds....")
                time.sleep(10)
                continue

            print(f"Camera: {self.camera_url} is working.......")
            window_name = f"Camera {self.camera_url}"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 400, 300)

            self._process_frames(
                capture,
                window_name,
                face_analyzer,
                face_matcher,
                region_detector,
                track_manager,
                image_manager,
            )

            capture.release()
            cv2.destroyWindow(window_name)

    def _process_frames(
        self,
        capture,
        window_name,
        face_analyzer,
        face_matcher,
        region_detector,
        track_manager,
        image_manager,
    ):
        """Process individual frames from video stream"""
        while not self.stop_event.is_set():
            # Skip frames
            for _ in range(self.skip_frames):
                capture.grab()

            ret, frame = capture.read()
            if not ret:
                print(
                    f"Video stream ended for {self.camera_url}. Monitoring for recovery."
                )
                break

            height, width, _ = frame.shape

            # Update regions if frame size changed
            block_regions = self.shared_data["block_regions"].get(self.ip_address)
            seat_regions = self.shared_data["seat_regions"].get(self.ip_address)
            region_detector.update_regions(block_regions, seat_regions, width, height)

            # Process faces
            faces = face_analyzer.get_faces(frame)
            if faces:
                self._process_faces(
                    frame,
                    faces,
                    width,
                    height,
                    face_matcher,
                    region_detector,
                    track_manager,
                    image_manager,
                )

            # Display frame
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) == ord("q"):
                print("Exit command received.")
                self.stop_event.set()
                break

    def _process_faces(
        self,
        frame,
        faces,
        width,
        height,
        face_matcher,
        region_detector,
        track_manager,
        image_manager,
    ):
        """Process detected faces in the frame"""
        copy_image = frame.copy()
        user_ids, feature_matrix = self.shared_data["features"]
        names_list = self.shared_data["names"]

        for face in faces:
            # Match face
            id_name, sim = face_matcher.match(
                face.normed_embedding, user_ids, feature_matrix
            )
            box = image_manager.get_coordinates(width, height, face.bbox)

            name = None
            if id_name != "Unknown":
                name = names_list[id_name]
                self._handle_recognized_face(
                    id_name,
                    name,
                    sim,
                    box,
                    width,
                    height,
                    copy_image,
                    region_detector,
                    track_manager,
                    image_manager,
                )

            # Draw bounding box
            image_manager.draw_bounding_box(frame, box, id_name, name, sim)

    def _handle_recognized_face(
        self,
        id_name,
        name,
        sim,
        box,
        width,
        height,
        copy_image,
        region_detector,
        track_manager,
        image_manager,
    ):
        """Handle logic for recognized faces"""
        last_recognition_time, last_camera = self.last_recognition_times.get(
            id_name, (0, None)
        )
        current_time = time.time()

        if current_time - last_recognition_time >= RECOGNITION_COOLDOWN or (
            self.ip_address != last_camera
        ):
            face_center = image_manager.get_face_center(box)

            # Check regions
            block_no = region_detector.check_block_region(copy_image, face_center)
            seat_no = region_detector.check_seat_region(copy_image, face_center)

            # Track if in any region or no regions defined
            if (
                block_no is not None
                or seat_no is not None
                or (
                    region_detector.block_points is None
                    and region_detector.seat_points is None
                )
            ):

                track_manager.mark_track_data(
                    id_name, self.ip_address, block_no, seat_no
                )
                self.last_recognition_times[id_name] = (current_time, self.ip_address)

                # Save face image
                face_image = image_manager.add_padding(copy_image, width, height, box)
                image_manager.save_track_face(face_image, id_name, name, sim)
