import sys
import torch
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QLineEdit, QPushButton, QHBoxLayout, QStackedLayout, QRadioButton
from PyQt5.QtGui import QPixmap, QImage, QFont
import cv2
from PyQt5.QtCore import Qt, QTimer
from my_detect import ObjectDetector
from datetime import datetime, timedelta
import numpy as np
import pyrealsense2 as rs

class VideoApp(QWidget):
    def __init__(self):
        super().__init__()

        self.detector = ObjectDetector()
        self.classNames = ["close_eyes", "close_mouth", "open_eyes", "open_mouth"]
        self.prev_mouth_state = -1
        self.prev_eye_state = -1
        self.frames_per_minute = -1
        self.blinkCount = 0
        self.yawnCount = 0
        self.setWindowTitle("HighAlert App")
        self.setGeometry(100, 100, 800, 600)

        self.video_widget = QWidget()

        self.video_layout = QVBoxLayout(self.video_widget)

        self.init_video_ui()

        self.stacked_layout = QStackedLayout(self)
        self.stacked_layout.addWidget(self.video_widget)

        self.stacked_layout.setCurrentIndex(0)

        self.start_time = None
        self.reset_time = 60  # Reset time in seconds
        self.score_increment = 1  # Score increment for each detection

        self.frame_counter = 0
        self.frames_per_minute = 0
        self.is_video_playing = False
        
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        self.pipeline.start(self.config)

    def init_video_ui(self):
        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)  # Fixed size for the video feed
        self.video_layout.addWidget(self.video_label)

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_video_feed)
        self.video_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_video_feed)
        self.video_layout.addWidget(self.stop_button)
        self.stop_button.setEnabled(False)

        self.score_label = QLabel("Score:")
        self.video_layout.addWidget(self.score_label)
        self.current_score = 0

        self.score_display = QLabel()
        font = QFont()
        font.setPointSize(24)  # Set font size to 24
        font.setBold(True)  # Set font bold
        self.score_display.setFont(font)
        self.score_display.setAlignment(Qt.AlignCenter)
        self.update_score_display()
        self.video_layout.addWidget(self.score_display)

        # Label for timer
        self.timer_label = QLabel("Timer: 00:00")
        self.video_layout.addWidget(self.timer_label)

        # Labels for displaying yawns and blinks
        self.yawn_label = QLabel("Yawns: 0")
        self.video_layout.addWidget(self.yawn_label)

        self.blink_label = QLabel("Blinks: 0")
        self.video_layout.addWidget(self.blink_label)

        self.selected_input_source = "video3.mp4"  # Default to webcam
        self.webcam_capture = cv2.VideoCapture(0)  # Use 0 for the default camera
        self.video_file = self.selected_input_source  # Path to video file
        self.video_capture = cv2.VideoCapture(self.video_file)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_video_feed)

    def start_video_feed(self):
        if not self.is_video_playing:
            self.is_video_playing = True
            self.start_time = datetime.now()
            self.timer.start(1)  # Update every 1 milliseconds
            self.start_button.setEnabled(False)  # Disable start button after starting the video feed
            self.stop_button.setEnabled(True)  # Enable stop button

    def stop_video_feed(self):
        if self.is_video_playing:
            self.is_video_playing = False
            self.timer.stop()  # Stop the video feed timer
            self.start_button.setEnabled(True)  # Enable start button
            self.stop_button.setEnabled(False)  # Disable stop button
            self.current_score = 0  # Reset the score counter
            self.update_score_display()  # Update the score display
            self.frame_counter = 0  # Reset the frame counter
            self.yawnCount = 0  # Reset yawn count
            self.blinkCount = 0  # Reset blink count

    def increase_score(self, score):
        self.current_score += score
        self.update_score_display()
        self.set_alert()

    def decrease_score(self, score):
        self.current_score -= score
        self.update_score_display()
        self.set_alert()

    def update_score_display(self):
        self.score_display.setText(str(self.current_score))

    def set_alert(self):
        # Check for score conditions
        if self.current_score > 10 and self.current_score <= 20:
            self.score_display.setStyleSheet("color: yellow")  # Yellow alert
            self.score_display.setText(f"Semi Unaware alert! Please be more alerted! Score: {self.current_score}")
        elif self.current_score > 20:
            self.score_display.setStyleSheet("color: red")  # Red alert
            self.score_display.setText(f"Fully Unaware alert!!! Please get up!!! Score: {self.current_score}")
        else:
            self.score_display.setStyleSheet("")  # No alert, revert to default color

    def update_video_feed(self):
        if self.is_video_playing:
            if self.selected_input_source == 'webcam':
                ret, frame = self.webcam_capture.read()
            else:
                ret, frame = self.video_capture.read()
                
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            frame = frames.get_color_frame()
            frame = np.asanyarray(frame.get_data())
        
            # Calculate time difference since start
            elapsed_time = (datetime.now() - self.start_time).total_seconds()
            minutes, seconds = divmod(elapsed_time, 60)
            timer_text = f"Timer: {int(minutes):02d}:{int(seconds):02d}"
            self.timer_label.setText(timer_text)

            # Check if it's time to reset counters
            if elapsed_time >= self.reset_time:
                self.calculate_score()
                self.start_time = datetime.now()

            # Check if it's time to reset counters after 5 minutes
            if elapsed_time >= 300:  # 5 minutes = 300 seconds
                self.start_time = datetime.now()
                self.current_score = 0  # Reset the score counter
                self.update_score_display()  # Update the score display
                self.frame_counter = 0  # Reset the frame counter
                self.yawnCount = 0  # Reset yawn count
                self.blinkCount = 0  # Reset blink count

            self.frame_counter += 1

            if ret:
                # Resize frame
                frame = cv2.resize(frame, (640, 480))

                # Detect objects in the frame
                detections = self.detector.detect(frame)

                # Draw bounding boxes and labels
                for detection in detections:
                    bbox = detection['bbox']
                    class_id = detection['class']
                    confidence = detection['confidence']
                    bbox[0] = int(bbox[0] * frame.shape[1] / self.detector.img_size)
                    bbox[1] = int(bbox[1] * frame.shape[0] / self.detector.img_size)
                    bbox[2] = int(bbox[2] * frame.shape[1] / self.detector.img_size)
                    bbox[3] = int(bbox[3] * frame.shape[0] / self.detector.img_size)
                    label = f'Class: {self.classNames[class_id]}, Confidence: {confidence:.2f}'
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                    cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Blink detection
                    #             open_eye             close_eye
                    if class_id == 2 and self.prev_eye_state == 0:
                        self.blinkCount += 1

                    # Yawn detection
                    #             close_mouth             open_mouth
                    elif class_id == 1 and self.prev_mouth_state == 3:
                        self.yawnCount += 1

                    # Update prev_states
                    if class_id == 1 or class_id == 3:
                        self.prev_mouth_state = class_id
                    elif class_id == 0 or class_id == 2:
                        self.prev_eye_state = class_id
                    else:
                        self.prev_eye_state = -1
                        self.prev_mouth_state = -1

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, channel = frame.shape
                bytes_per_line = 3 * width
                q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_image)
                self.video_label.setPixmap(pixmap)

                # Update yawn and blink labels
                self.yawn_label.setText(f"Yawns: {self.yawnCount}")
                self.blink_label.setText(f"Blinks: {self.blinkCount}")

    def calculate_score(self):
        print("###############################################################")
        print("Print Results:")
        # Calculate frames per minute
        self.frames_per_minute = int(self.frame_counter / (self.reset_time / 60))
        print("Frames per minute:", self.frames_per_minute)

        self.blinkCountPerMinute = self.blinkCount
        self.frame_counter = 0

        # Blinking
        if self.blinkCountPerMinute == 0:  # eyes closed
            self.increase_score(20)
        if 0 < self.blinkCountPerMinute <= 4:
            self.increase_score(2)
        if 4< self.blinkCountPerMinute <= 8:
            self.increase_score(6)
        if 8 < self.blinkCountPerMinute:
            self.increase_score(10)

        # Yawning
        if 0 <= self.yawnCount <= 2:
            self.increase_score(3)
        if self.yawnCount > 2:
            self.increase_score(10)

        print("Yawns: " + str(self.yawnCount))
        print("Blinks: " + str(self.blinkCount))
        print("Score: " + str(self.current_score))

        print("###############################################################")

        # Reset class counters to zero
        self.blinkCount = 0
        self.yawnCount = 0

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoApp()
    window.show()
    sys.exit(app.exec_())
