import os
import cv2
import numpy as np
import pyttsx3
import time
from ultralytics import YOLO
import threading
from collections import deque


# Function to play text-to-speech in a separate thread
def speak_in_thread(text, engine):
    print(text)
    t = threading.Thread(target=lambda: speak_now(text, engine))
    t.daemon = True
    t.start()


def speak_now(text, engine):
    engine.say(text)
    engine.runAndWait()


# Function to calculate focal length
def calculate_focal_length(known_width, known_distance, perceived_width):
    return (perceived_width * known_distance) / known_width


# Function to calculate distance
def calculate_distance(known_width, focal_length, perceived_width):
    return (known_width * focal_length) / perceived_width


# Function to get object position (left, center, right)
def get_position(centroid_x, frame_width):
    if centroid_x < frame_width / 3:
        return "on your left"
    elif centroid_x < 2 * frame_width / 3:
        return "ahead of you"
    else:
        return "on your right"


class ObjectTracker:
    def __init__(self, history_size=10):
        self.history_size = history_size
        self.distance_history = deque(maxlen=history_size)
        self.time_history = deque(maxlen=history_size)
        self.position_history = deque(maxlen=history_size)
        self.speed = 0
        self.direction = ""
        self.last_update = 0

    def update(self, distance, position, current_time):
        """Update tracker with new measurements"""
        self.distance_history.append(distance)
        self.time_history.append(current_time)
        self.position_history.append(position)
        self.last_update = current_time

        # Calculate speed using multiple samples
        if len(self.distance_history) >= 3:
            self._calculate_speed()
            self._calculate_direction()

    def _calculate_speed(self):
        """Calculate speed using linear regression for smoothing"""
        if len(self.distance_history) < 3:
            self.speed = 0
            return

        # Use the last 5 samples or all available if fewer
        samples = min(5, len(self.distance_history))

        distances = list(self.distance_history)[-samples:]
        times = list(self.time_history)[-samples:]

        # Calculate time differences from the first sample
        time_diffs = [t - times[0] for t in times]

        if max(time_diffs) < 0.1:  # Ensure we have enough time difference
            self.speed = 0
            return

        # Simple linear regression to find speed
        try:
            if len(set(time_diffs)) < 2:  # Avoid division by zero
                self.speed = 0
                return

            # Use numpy for linear regression
            slope, _ = np.polyfit(time_diffs, distances, 1)

            # Absolute value of slope is the speed
            self.speed = abs(slope)

            # Smooth sudden changes
            if hasattr(self, 'prev_speed') and self.prev_speed is not None:
                self.speed = 0.7 * self.speed + 0.3 * self.prev_speed

            self.prev_speed = self.speed
        except:
            self.speed = 0

    def _calculate_direction(self):
        """Determine movement direction based on position history"""
        if len(self.position_history) < 3:
            self.direction = ""
            return

        # Get last 3 positions
        positions = list(self.position_history)[-3:]
        x_coords = [p[0] for p in positions]
        y_coords = [p[1] for p in positions]

        # Calculate average movements
        x_diff = x_coords[-1] - x_coords[0]
        y_diff = y_coords[-1] - y_coords[0]

        # Determine dominant direction
        if abs(x_diff) > abs(y_diff):
            if x_diff > 10:
                self.direction = "moving right"
            elif x_diff < -10:
                self.direction = "moving left"
            else:
                self.direction = ""
        else:
            if y_diff > 10:
                self.direction = "moving toward you"
            elif y_diff < -10:
                self.direction = "moving away"
            else:
                self.direction = ""

    def get_speed(self):
        """Return current speed estimate"""
        return self.speed

    def get_direction(self):
        """Return current direction estimate"""
        return self.direction

    def is_stale(self, current_time, timeout=2.0):
        """Check if this tracker hasn't been updated recently"""
        return (current_time - self.last_update) > timeout


# Dictionary to store class-specific widths (in cm)
OBJECT_WIDTHS = {
    "person": 50,
    "bicycle": 60,
    "car": 180,
    "motorcycle": 80,
    "bus": 250,
    "truck": 260,
    "chair": 40,
    "dining table": 120,
    "laptop": 35,
    "remote": 15,
    "keyboard": 30,
    "cell phone": 8,
    "book": 20,
    "bottle": 8,
    "cup": 8,
    "bowl": 15
}

# Default width for unknown objects
DEFAULT_WIDTH = 30


# Main function
def main():
    # Path to YOLO
    #
    # v8 model
    model_path = "yolov8n.pt"  # Use smallest YOLOv8 model for speed

    # Load YOLOv8 model with GPU if available
    model = YOLO(model_path)

    # Initialize video capture
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        raise RuntimeError("Error: Could not open video stream.")

    # Set lower resolution for faster processing
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Try to set a fixed frame rate
    video.set(cv2.CAP_PROP_FPS, 30)

    # Initialize TTS engine
    engine = pyttsx3.init()
    engine.setProperty('rate', 180)  # Slightly faster speech
    engine.setProperty('volume', 1)
    speak_in_thread("Object detection system activated. Press Q to quit.", engine)

    # Calibration: Use a default focal length instead of calibration for demo
    focal_length = 800  # Default focal length assumption

    speak_in_thread("Using default calibration. Press C to calibrate with an object.", engine)

    # Object trackers
    trackers = {}

    last_spoken_time = {}
    last_announcements = {}
    cooldown_period = 3  # Reduced cooldown period
    skip_frames = 1  # Process every frame for better speed calculation
    frame_count = 0

    # Lower the speed threshold to include more movement announcements
    speed_threshold = 5  # cm/s - now announce speeds above this threshold

    # Clean up stale trackers periodically
    cleanup_interval = 30  # frames

    # Frame timing for more accurate speed calculations
    prev_frame_time = time.time()

    while True:
        # Calculate actual frame time
        current_frame_time = time.time()
        frame_time = current_frame_time - prev_frame_time
        prev_frame_time = current_frame_time

        # Limit processing rate if running too fast
        if frame_time < 0.01:  # Cap at 100fps
            time.sleep(0.01 - frame_time)

        ret, frame = video.read()
        if not ret:
            speak_in_thread("Error: Failed to capture frame.", engine)
            break

        frame_count += 1
        if frame_count % skip_frames != 0:  # Skip frames for performance
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            continue

        # Current time for calculations
        current_time = time.time()

        # Clean up stale trackers
        if frame_count % cleanup_interval == 0:
            stale_keys = [k for k in trackers.keys() if trackers[k].is_stale(current_time)]
            for k in stale_keys:
                del trackers[k]

        # Run object detection on the frame
        results = model(frame, conf=0.5)  # Increased confidence threshold

        # Create a combined announcement
        announcements = []

        # Process detected objects
        detected_objects = set()

        for result in results:
            for box in result.boxes:
                x, y, w, h = box.xywh[0].tolist()
                label = model.names[int(box.cls)]
                confidence = float(box.conf)
                centroid_x, centroid_y = int(x), int(y)

                # Skip small detections (likely false positives)
                if w * h < 1000:
                    continue

                # Store this as a detected object
                detected_objects.add(label)

                # Look up width for this object or use default
                known_width = OBJECT_WIDTHS.get(label, DEFAULT_WIDTH)

                # Calculate distance using object-specific width
                distance = calculate_distance(known_width, focal_length, w)

                # Unique tracker ID
                tracker_id = f"{label}"

                # Create a new tracker if we haven't seen this object
                if tracker_id not in trackers:
                    trackers[tracker_id] = ObjectTracker()

                # Update the tracker with new measurements
                trackers[tracker_id].update(distance, (centroid_x, centroid_y), current_time)

                # Get speed and direction from tracker
                speed = trackers[tracker_id].get_speed()
                direction = trackers[tracker_id].get_direction()

                # Position description
                position = get_position(centroid_x, frame.shape[1])

                # Prepare announcement
                if tracker_id not in last_spoken_time or (
                        current_time - last_spoken_time[tracker_id]) > cooldown_period:
                    announcement = f"{label} {position}, {distance:.1f} centimeters away"

                    # Add direction information if available
                    if direction:
                        announcement += f", {direction}"

                    # Always add speed information
                    if speed < 2:
                        announcement += ", stationary"
                    else:
                        # Convert to a human-readable speed description
                        if speed < 20:
                            speed_desc = "slowly"
                        elif speed < 60:
                            speed_desc = "moderately"
                        else:
                            speed_desc = "quickly"

                        announcement += f", moving {speed_desc} at {speed:.1f} centimeters per second"

                    announcements.append(announcement)
                    last_spoken_time[tracker_id] = current_time
                    last_announcements[tracker_id] = announcement

                # Draw bounding box and label
                cv2.rectangle(frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0, 255, 0), 2)

                # Always show speed and distance on display
                label_text = f"{label}: {distance:.1f}cm, {speed:.1f} cm/s"
                if direction:
                    label_text += f", {direction}"

                cv2.putText(frame, label_text, (int(x - w / 2), int(y - h / 2) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Make a single announcement with all detected objects
        if announcements:
            if len(announcements) == 1:
                speak_in_thread(f"I found: {announcements[0]}", engine)
            else:
                combined = "I found: " + ", and ".join([", ".join(announcements[:-1]), announcements[-1]])
                speak_in_thread(combined, engine)

        # Display the frame
        cv2.imshow("Object Detection", frame)

        # Check for user input
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):  # Quit
            break
        elif key == ord(" "):  # Repeat last announcement
            if last_announcements:
                combined = "Last detected: " + ", ".join(last_announcements.values())
                speak_in_thread(combined, engine)
        elif key == ord("c"):  # Calibrate
            speak_in_thread("Calibration mode. Please place a standard object 1 meter away and press space.", engine)
            calibrating = True
            while calibrating:
                ret, cal_frame = video.read()
                if not ret:
                    break

                cv2.putText(cal_frame, "CALIBRATION MODE - Place object 1m away", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("Object Detection", cal_frame)

                cal_key = cv2.waitKey(1) & 0xFF
                if cal_key == ord(" "):
                    # Detect objects in calibration frame
                    cal_results = model(cal_frame, conf=0.6)
                    if cal_results and len(cal_results[0].boxes) > 0:
                        # Use the largest object for calibration
                        areas = [box.xywh[0][2] * box.xywh[0][3] for box in cal_results[0].boxes]
                        largest_idx = np.argmax(areas)
                        cal_box = cal_results[0].boxes[largest_idx]
                        cal_label = model.names[int(cal_box.cls)]
                        cal_width = OBJECT_WIDTHS.get(cal_label, DEFAULT_WIDTH)
                        cal_perceived_width = cal_box.xywh[0][2].item()

                        # Calculate focal length
                        focal_length = calculate_focal_length(cal_width, 100, cal_perceived_width)
                        speak_in_thread(f"Calibrated with {cal_label}, focal length: {focal_length:.1f}", engine)
                    else:
                        speak_in_thread("No object detected. Calibration failed.", engine)
                    calibrating = False
                elif cal_key == ord("q"):
                    calibrating = False

    # Release resources
    video.release()
    cv2.destroyAllWindows()
    speak_in_thread("Object detection system deactivated.", engine)


if __name__ == "__main__":
    main()