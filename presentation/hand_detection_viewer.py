"""
Hand detection viewer using OpenCV
"""
import cv2
import numpy as np
from typing import Optional
from domain.interfaces import IHandDetector, ICamera
from domain.models import DetectionResult, Hand, Point


class HandDetectionViewer:
    """Viewer for hand detection results"""
    
    def __init__(self, detector: IHandDetector, camera: ICamera):
        """
        Initialize viewer
        
        Args:
            detector: Hand detector instance
            camera: Camera instance
        """
        self.detector = detector
        self.camera = camera
        self.window_name = "Hand Detection - MediaPipe"
    
    def draw_landmarks(self, image: np.ndarray, detection_result: DetectionResult) -> np.ndarray:
        """
        Draw hand landmarks and connections on image
        
        Args:
            image: Input image
            detection_result: Detection result
            
        Returns:
            Image with landmarks drawn
        """
        if not detection_result or not detection_result.hands:
            return image
        
        annotated_image = image.copy()
        h, w, _ = annotated_image.shape
        
        # Hand connections (MediaPipe format)
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (5, 9), (9, 13), (13, 17)  # Palm connections
        ]
        
        for hand in detection_result.hands:
            # Draw connections
            for start_idx, end_idx in connections:
                start_point = hand.landmarks[start_idx]
                end_point = hand.landmarks[end_idx]
                
                start = (int(start_point.x * w), int(start_point.y * h))
                end = (int(end_point.x * w), int(end_point.y * h))
                
                cv2.line(annotated_image, start, end, (0, 255, 0), 2)
            
            # Draw landmarks
            for idx, landmark in enumerate(hand.landmarks):
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                
                # Different colors for different landmark types
                if idx == 0:  # Wrist
                    color = (255, 0, 255)  # Magenta
                    radius = 5
                elif idx in [4, 8, 12, 16, 20]:  # Fingertips
                    color = (0, 0, 255)  # Red
                    radius = 4
                else:
                    color = (0, 255, 0)  # Green
                    radius = 3
                
                cv2.circle(annotated_image, (x, y), radius, color, -1)
            
            # Draw handedness label
            wrist = hand.landmarks[0]
            label_x = int(wrist.x * w)
            label_y = int(wrist.y * h) - 20
            label = f"{hand.handedness} ({hand.confidence:.2f})"
            
            cv2.putText(annotated_image, label, (label_x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(annotated_image, label, (label_x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return annotated_image
    
    def draw_info(self, image: np.ndarray, fps: float, num_hands: int) -> np.ndarray:
        """
        Draw FPS and hand count info
        
        Args:
            image: Input image
            fps: Frames per second
            num_hands: Number of detected hands
            
        Returns:
            Image with info overlay
        """
        annotated_image = image.copy()
        
        # FPS
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(annotated_image, fps_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Hand count
        hands_text = f"Hands: {num_hands}"
        cv2.putText(annotated_image, hands_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Instructions
        instruction = "Press 'q' to quit"
        cv2.putText(annotated_image, instruction, (10, annotated_image.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated_image
    
    def run(self) -> None:
        """Run the hand detection viewer"""
        # Initialize detector
        if not self.detector.initialize():
            print("Failed to initialize hand detector")
            return
        
        # Open camera
        if not self.camera.open():
            print("Failed to open camera")
            self.detector.release()
            return
        
        print("Hand Detection started. Press 'q' to quit.")
        
        import time
        fps_start_time = time.time()
        fps_frame_count = 0
        fps = 0.0
        
        try:
            while True:
                # Read frame
                result = self.camera.read()
                if result is None:
                    break
                
                success, frame = result
                if not success:
                    break
                
                # Detect hands
                detection_result = self.detector.detect(frame)
                
                # Draw landmarks
                if detection_result:
                    frame = self.draw_landmarks(frame, detection_result)
                    num_hands = len(detection_result.hands)
                else:
                    num_hands = 0
                
                # Calculate FPS
                fps_frame_count += 1
                if fps_frame_count >= 30:
                    fps = 30.0 / (time.time() - fps_start_time)
                    fps_start_time = time.time()
                    fps_frame_count = 0
                
                # Draw info
                frame = self.draw_info(frame, fps, num_hands)
                
                # Display frame
                cv2.imshow(self.window_name, frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Cleanup
            cv2.destroyAllWindows()
            self.camera.release()
            self.detector.release()
            print("Hand Detection stopped.")

