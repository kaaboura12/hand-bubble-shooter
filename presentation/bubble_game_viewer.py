"""
Bubble game viewer with hand detection - Two separate windows
"""
import cv2
import numpy as np
import time
from typing import Optional, Tuple
from domain.interfaces import IHandDetector, ICamera
from domain.models import DetectionResult, Hand, Point, HandLandmark
from domain.gesture_detector import GestureDetector
from presentation.bubble_game import BubbleGame


class BubbleGameViewer:
    """Viewer for bubble game with hand detection - Combined single window"""
    
    def __init__(self, detector: IHandDetector, camera: ICamera):
        """
        Initialize bubble game viewer
        
        Args:
            detector: Hand detector instance
            camera: Camera instance
        """
        self.detector = detector
        self.camera = camera
        self.window_name = "Bubble Game - Hand Detection"
        self.game: Optional[BubbleGame] = None
        self.last_update_time = time.time()
        self.gesture_detector = GestureDetector()
        self.player_name: Optional[str] = None
        self.camera_view_size = (200, 150)  # Width, Height for top right corner
    
    def draw_hand_landmarks(self, image: np.ndarray, detection_result: DetectionResult) -> np.ndarray:
        """
        Draw full hand landmarks on camera feed
        
        Args:
            image: Input image
            detection_result: Detection result
            
        Returns:
            Image with hand landmarks drawn
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
            
            # Draw shooting status on camera
            is_closed = self.gesture_detector.is_hand_closed(hand)
            status_text = "CLOSED (SHOOTING)" if is_closed else "OPEN (AIMING)"
            status_color = (0, 0, 255) if is_closed else (0, 255, 0)
            wrist = hand.landmarks[0]
            label_x = int(wrist.x * w)
            label_y = int(wrist.y * h) - 30
            cv2.putText(annotated_image, status_text, (label_x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        return annotated_image
    
    def draw_camera_view(self, main_image: np.ndarray, camera_frame: np.ndarray) -> np.ndarray:
        """
        Draw camera view in top right corner of main image
        
        Args:
            main_image: Main game image
            camera_frame: Camera frame with hand detection
            
        Returns:
            Combined image
        """
        combined_image = main_image.copy()
        h, w = combined_image.shape[:2]
        
        # Camera view size
        cam_w, cam_h = self.camera_view_size
        
        # Resize camera frame to fit in corner
        camera_resized = cv2.resize(camera_frame, (cam_w, cam_h))
        
        # Position in top right corner
        x_offset = w - cam_w - 10
        y_offset = 10
        
        # Draw border
        cv2.rectangle(combined_image,
                     (x_offset - 2, y_offset - 2),
                     (x_offset + cam_w + 2, y_offset + cam_h + 2),
                     (255, 255, 255), 2)
        
        # Overlay camera view
        combined_image[y_offset:y_offset + cam_h, x_offset:x_offset + cam_w] = camera_resized
        
        # Draw label
        label = "Camera"
        cv2.putText(combined_image, label, (x_offset, y_offset - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return combined_image
    
    def create_game_screen(self, width: int, height: int) -> np.ndarray:
        """
        Create game screen with bubbles and pointer
        
        Args:
            width: Screen width
            height: Screen height
            pointer_x: Pointer x position (normalized)
            pointer_y: Pointer y position (normalized)
            is_shooting: True if shooting
            
        Returns:
            Game screen image
        """
        # Create black background
        game_screen = np.zeros((height, width, 3), dtype=np.uint8)
        return game_screen
    
    def draw_bubbles(self, image: np.ndarray) -> np.ndarray:
        """
        Draw bubbles on game screen
        
        Args:
            image: Game screen image
            
        Returns:
            Image with bubbles drawn
        """
        if not self.game:
            return image
        
        annotated_image = image.copy()
        h, w = annotated_image.shape[:2]
        
        for bubble in self.game.bubbles:
            x = int(bubble.x * w)
            y = int(bubble.y * h)
            
            # Draw bubble with smooth edges
            cv2.circle(annotated_image, (x, y), bubble.radius, bubble.color, -1)
            
            # Outer border
            cv2.circle(annotated_image, (x, y), bubble.radius, (255, 255, 255), 2)
            
            # Inner border for depth
            cv2.circle(annotated_image, (x, y), bubble.radius - 2, (200, 200, 200), 1)
            
            # Draw highlight for 3D effect
            highlight_x = x - bubble.radius // 3
            highlight_y = y - bubble.radius // 3
            highlight_radius = bubble.radius // 3
            cv2.circle(annotated_image, (highlight_x, highlight_y), 
                      highlight_radius, (255, 255, 255), -1)
            
            # Small inner highlight
            cv2.circle(annotated_image, (highlight_x, highlight_y), 
                      highlight_radius // 2, (255, 255, 255), -1)
        
        return annotated_image
    
    def draw_pointer(self, image: np.ndarray, pointer_x: float, pointer_y: float, 
                     is_shooting: bool) -> np.ndarray:
        """
        Draw crosshair/pointer on game screen
        
        Args:
            image: Game screen image
            pointer_x: Pointer x position (normalized 0.0 to 1.0)
            pointer_y: Pointer y position (normalized 0.0 to 1.0)
            is_shooting: True if hand is closed (shooting)
            
        Returns:
            Image with pointer drawn
        """
        annotated_image = image.copy()
        h, w = annotated_image.shape[:2]
        
        x = int(pointer_x * w)
        y = int(pointer_y * h)
        
        # Color changes based on shooting state
        if is_shooting:
            color = (0, 0, 255)  # Red when shooting
            size = 30
        else:
            color = (0, 255, 255)  # Yellow when aiming
            size = 25
        
        # Draw crosshair with anti-aliasing effect
        line_length = size
        thickness = 3
        
        # Outer glow effect (lighter version)
        glow_color = tuple(min(255, c + 50) for c in color)
        cv2.line(annotated_image, 
                (x - line_length - 2, y), 
                (x + line_length + 2, y), 
                glow_color, 1)
        cv2.line(annotated_image, 
                (x, y - line_length - 2), 
                (x, y + line_length + 2), 
                glow_color, 1)
        
        # Main crosshair lines
        cv2.line(annotated_image, 
                (x - line_length, y), 
                (x + line_length, y), 
                color, thickness)
        cv2.line(annotated_image, 
                (x, y - line_length), 
                (x, y + line_length), 
                color, thickness)
        
        # Center circle with glow
        cv2.circle(annotated_image, (x, y), size // 2 + 1, glow_color, 1)
        cv2.circle(annotated_image, (x, y), size // 2, color, 2)
        cv2.circle(annotated_image, (x, y), 4, color, -1)
        
        return annotated_image
    
    def draw_game_info(self, image: np.ndarray, fps: float, is_shooting: bool) -> np.ndarray:
        """
        Draw game information on game screen
        
        Args:
            image: Game screen image
            fps: Frames per second
            is_shooting: True if hand is closed (shooting)
            
        Returns:
            Image with game info
        """
        if not self.game:
            return image
        
        annotated_image = image.copy()
        
        # Player name
        if self.player_name:
            name_text = f"Player: {self.player_name}"
            cv2.putText(annotated_image, name_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated_image, name_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            y_offset = 60
        else:
            y_offset = 30
        
        # Score
        score_text = f"Score: {self.game.score}"
        cv2.putText(annotated_image, score_text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(annotated_image, score_text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
        
        # Bubbles popped
        popped_text = f"Popped: {self.game.bubbles_popped}"
        cv2.putText(annotated_image, popped_text, (10, y_offset + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(annotated_image, popped_text, (10, y_offset + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        # Active bubbles
        bubbles_text = f"Bubbles: {len(self.game.bubbles)}"
        cv2.putText(annotated_image, bubbles_text, (10, y_offset + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(annotated_image, bubbles_text, (10, y_offset + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        # Shooting status
        shoot_status = "SHOOTING!" if is_shooting else "Aiming..."
        shoot_color = (0, 0, 255) if is_shooting else (0, 255, 255)
        cv2.putText(annotated_image, shoot_status, (10, y_offset + 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, shoot_color, 2)
        cv2.putText(annotated_image, shoot_status, (10, y_offset + 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        # FPS
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(annotated_image, fps_text, (10, y_offset + 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Instructions
        instruction = "Press 'q' to quit, 'r' to reset"
        cv2.putText(annotated_image, instruction, (10, annotated_image.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(annotated_image, instruction, (10, annotated_image.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return annotated_image
    
    def run(self) -> None:
        """Run the bubble game with combined view in single window"""
        # Initialize detector
        if not self.detector.initialize():
            print("Failed to initialize hand detector")
            return
        
        # Open camera
        if not self.camera.open():
            print("Failed to open camera")
            self.detector.release()
            return
        
        # Get frame dimensions
        result = self.camera.read()
        if result is None:
            print("Failed to read from camera")
            self.camera.release()
            self.detector.release()
            return
        
        _, camera_frame = result
        camera_h, camera_w = camera_frame.shape[:2]
        
        # Game screen dimensions - full screen minus camera view space
        game_width = 800
        game_height = 600
        
        # Initialize game with balanced settings
        self.game = BubbleGame(
            screen_width=game_width,
            screen_height=game_height,
            max_bubbles=5,  # Reduced for better gameplay
            spawn_rate=0.8,  # Slower spawn rate
            min_radius=30,
            max_radius=50
        )
        
        print("=" * 60)
        print("Bubble Game Started!")
        print("Single window with camera view in top right corner")
        print("Aim with your index finger, close hand to shoot!")
        print("Press 'q' to quit, 'r' to reset score")
        print("=" * 60)
        
        fps_start_time = time.time()
        fps_frame_count = 0
        fps = 0.0
        self.last_update_time = time.time()
        
        # Track pointer position with smoothing
        pointer_x = 0.5
        pointer_y = 0.5
        target_pointer_x = 0.5
        target_pointer_y = 0.5
        smoothing_factor = 0.15  # Lower = smoother but slower response
        is_shooting = False
        
        try:
            while True:
                current_time = time.time()
                delta_time = current_time - self.last_update_time
                self.last_update_time = current_time
                
                # Read frame
                result = self.camera.read()
                if result is None:
                    break
                
                success, camera_frame = result
                if not success:
                    break
                
                # Update game (clamp delta_time for stability)
                if self.game:
                    clamped_delta = min(delta_time, 0.1)  # Prevent large time jumps
                    self.game.update(clamped_delta)
                
                # Detect hands
                detection_result = self.detector.detect(camera_frame)
                
                # Update pointer and shooting state with smoothing
                if detection_result and detection_result.hands:
                    # Use first detected hand
                    hand = detection_result.hands[0]
                    
                    # Get index finger tip for pointer
                    finger_tip = self.gesture_detector.get_index_finger_tip(hand)
                    if finger_tip:
                        # Update target position
                        target_pointer_x = finger_tip.x
                        target_pointer_y = finger_tip.y
                        
                        # Smooth interpolation for pointer movement
                        pointer_x += (target_pointer_x - pointer_x) * smoothing_factor
                        pointer_y += (target_pointer_y - pointer_y) * smoothing_factor
                    
                    # Check if hand is closed (shooting)
                    is_shooting = self.gesture_detector.is_hand_closed(hand)
                    
                    # Check collisions when shooting (use smoothed position)
                    if is_shooting and self.game:
                        self.game.check_collisions(pointer_x, pointer_y, is_shooting)
                else:
                    is_shooting = False
                    # Smoothly return to center when no hand detected
                    target_pointer_x = 0.5
                    target_pointer_y = 0.5
                    pointer_x += (target_pointer_x - pointer_x) * smoothing_factor
                    pointer_y += (target_pointer_y - pointer_y) * smoothing_factor
                
                # ===== CREATE COMBINED VIEW =====
                # Create game screen
                game_screen = self.create_game_screen(game_width, game_height)
                
                # Draw bubbles
                game_screen = self.draw_bubbles(game_screen)
                
                # Draw pointer
                game_screen = self.draw_pointer(game_screen, pointer_x, pointer_y, is_shooting)
                
                # Draw game info
                game_screen = self.draw_game_info(game_screen, fps, is_shooting)
                
                # Draw camera view with hand landmarks
                camera_display = self.draw_hand_landmarks(camera_frame.copy(), detection_result)
                
                # Combine: camera in top right, game fills rest
                combined_view = self.draw_camera_view(game_screen, camera_display)
                
                # Calculate FPS
                fps_frame_count += 1
                if fps_frame_count >= 30:
                    fps = 30.0 / (time.time() - fps_start_time)
                    fps_start_time = time.time()
                    fps_frame_count = 0
                
                # Display combined window
                cv2.imshow(self.window_name, combined_view)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r') and self.game:
                    self.game.reset()
                    print("Game reset!")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Cleanup
            cv2.destroyAllWindows()
            self.camera.release()
            self.detector.release()
            if self.game:
                print(f"\nFinal Score: {self.game.score}")
                print(f"Bubbles Popped: {self.game.bubbles_popped}")
            print("Bubble Game stopped.")
