"""
Menu system with hand gesture navigation
"""
import cv2
import numpy as np
import time
from typing import Optional, List, Tuple, Callable
from domain.interfaces import IHandDetector, ICamera
from domain.models import DetectionResult, Hand, Point
from domain.gesture_detector import GestureDetector


class MenuItem:
    """Represents a menu item"""
    def __init__(self, text: str, action: Callable, position: Tuple[int, int]):
        self.text = text
        self.action = action
        self.position = position  # (x, y) center position
        self.width = 200
        self.height = 50
        self.is_hovered = False


class MenuSystem:
    """Menu system with hand gesture navigation"""
    
    def __init__(self, detector: IHandDetector, camera: ICamera):
        """
        Initialize menu system
        
        Args:
            detector: Hand detector instance
            camera: Camera instance
        """
        self.detector = detector
        self.camera = camera
        self.window_name = "Menu - Hand Detection"
        self.gesture_detector = GestureDetector()
        self.menu_items: List[MenuItem] = []
        self.selected_index = 0
        self.last_selection_time = 0
        self.selection_cooldown = 1.0  # Seconds between selections
        
    def add_menu_item(self, text: str, action: Callable, position: Tuple[int, int]):
        """Add a menu item"""
        item = MenuItem(text, action, position)
        self.menu_items.append(item)
    
    def draw_menu(self, image: np.ndarray) -> np.ndarray:
        """
        Draw menu on image
        
        Args:
            image: Input image
            
        Returns:
            Image with menu drawn
        """
        annotated_image = image.copy()
        h, w = annotated_image.shape[:2]
        
        # Draw title
        title = "Bubble Game"
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        title_x = (w - title_size[0]) // 2
        title_y = 80
        
        cv2.putText(annotated_image, title, (title_x, title_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(annotated_image, title, (title_x, title_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 1)
        
        # Draw subtitle
        subtitle = "Use your finger to navigate"
        subtitle_size = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        subtitle_x = (w - subtitle_size[0]) // 2
        subtitle_y = title_y + 40
        
        cv2.putText(annotated_image, subtitle, (subtitle_x, subtitle_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # Draw menu items
        for idx, item in enumerate(self.menu_items):
            x, y = item.position
            is_selected = (idx == self.selected_index)
            
            # Background rectangle
            if is_selected:
                color = (100, 200, 255)  # Light blue when selected
                border_color = (255, 255, 0)  # Yellow border
                border_thickness = 3
            else:
                color = (50, 50, 50)  # Dark gray
                border_color = (150, 150, 150)  # Light gray border
                border_thickness = 2
            
            # Draw background
            top_left = (x - item.width // 2, y - item.height // 2)
            bottom_right = (x + item.width // 2, y + item.height // 2)
            cv2.rectangle(annotated_image, top_left, bottom_right, color, -1)
            cv2.rectangle(annotated_image, top_left, bottom_right, border_color, border_thickness)
            
            # Draw text
            text_size = cv2.getTextSize(item.text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = x - text_size[0] // 2
            text_y = y + text_size[1] // 2
            
            text_color = (255, 255, 255) if is_selected else (200, 200, 200)
            cv2.putText(annotated_image, item.text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        
        # Draw instructions
        instruction = "Point at menu item and close hand to select"
        cv2.putText(annotated_image, instruction, (10, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated_image
    
    def draw_hand_landmarks(self, image: np.ndarray, detection_result: DetectionResult) -> np.ndarray:
        """Draw hand landmarks"""
        if not detection_result or not detection_result.hands:
            return image
        
        annotated_image = image.copy()
        h, w, _ = annotated_image.shape
        
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
                if start_idx < len(hand.landmarks) and end_idx < len(hand.landmarks):
                    start_point = hand.landmarks[start_idx]
                    end_point = hand.landmarks[end_idx]
                    
                    start = (int(start_point.x * w), int(start_point.y * h))
                    end = (int(end_point.x * w), int(end_point.y * h))
                    
                    cv2.line(annotated_image, start, end, (0, 255, 0), 2)
            
            # Draw index finger tip prominently
            if len(hand.landmarks) > 8:
                finger_tip = hand.landmarks[8]
                tip_x = int(finger_tip.x * w)
                tip_y = int(finger_tip.y * h)
                cv2.circle(annotated_image, (tip_x, tip_y), 10, (0, 255, 255), -1)
                cv2.circle(annotated_image, (tip_x, tip_y), 10, (0, 0, 255), 2)
        
        return annotated_image
    
    def check_menu_selection(self, detection_result: DetectionResult) -> Optional[int]:
        """
        Check if user selected a menu item
        
        Returns:
            Index of selected item or None
        """
        if not detection_result or not detection_result.hands:
            return None
        
        current_time = time.time()
        if current_time - self.last_selection_time < self.selection_cooldown:
            return None
        
        hand = detection_result.hands[0]
        
        # Check if hand is closed (selection gesture)
        is_closed = self.gesture_detector.is_hand_closed(hand)
        if not is_closed:
            return None
        
        # Get index finger tip position
        finger_tip = self.gesture_detector.get_index_finger_tip(hand)
        if not finger_tip:
            return None
        
        # Get frame dimensions
        result = self.camera.read()
        if result is None:
            return None
        
        _, frame = result
        h, w = frame.shape[:2]
        
        finger_x = int(finger_tip.x * w)
        finger_y = int(finger_tip.y * h)
        
        # Check which menu item is being pointed at
        for idx, item in enumerate(self.menu_items):
            x, y = item.position
            item_left = x - item.width // 2
            item_right = x + item.width // 2
            item_top = y - item.height // 2
            item_bottom = y + item.height // 2
            
            if (item_left <= finger_x <= item_right and 
                item_top <= finger_y <= item_bottom):
                self.last_selection_time = current_time
                return idx
        
        return None
    
    def update_selection(self, detection_result: DetectionResult):
        """Update which menu item is being hovered"""
        if not detection_result or not detection_result.hands:
            return
        
        hand = detection_result.hands[0]
        finger_tip = self.gesture_detector.get_index_finger_tip(hand)
        if not finger_tip:
            return
        
        result = self.camera.read()
        if result is None:
            return
        
        _, frame = result
        h, w = frame.shape[:2]
        
        finger_x = int(finger_tip.x * w)
        finger_y = int(finger_tip.y * h)
        
        # Find which menu item is being pointed at
        for idx, item in enumerate(self.menu_items):
            x, y = item.position
            item_left = x - item.width // 2
            item_right = x + item.width // 2
            item_top = y - item.height // 2
            item_bottom = y + item.height // 2
            
            if (item_left <= finger_x <= item_right and 
                item_top <= finger_y <= item_bottom):
                self.selected_index = idx
                break
    
    def run(self) -> Optional[str]:
        """
        Run menu system
        
        Returns:
            Selected action result (e.g., "start_game") or None
        """
        if not self.detector.initialize():
            print("Failed to initialize hand detector")
            return None
        
        if not self.camera.open():
            print("Failed to open camera")
            self.detector.release()
            return None
        
        print("Menu system started. Point at menu items and close hand to select.")
        
        try:
            while True:
                result = self.camera.read()
                if result is None:
                    break
                
                success, frame = result
                if not success:
                    break
                
                # Detect hands
                detection_result = self.detector.detect(frame)
                
                # Update selection
                if detection_result:
                    self.update_selection(detection_result)
                
                # Check for selection
                selected_idx = self.check_menu_selection(detection_result)
                if selected_idx is not None:
                    item = self.menu_items[selected_idx]
                    print(f"Selected: {item.text}")
                    result = item.action()
                    if result:
                        return result
                
                # Draw menu
                frame = self.draw_menu(frame)
                frame = self.draw_hand_landmarks(frame, detection_result)
                
                # Display
                cv2.imshow(self.window_name, frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return None
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            cv2.destroyAllWindows()
            # Don't release camera here - it will be reused
        
        return None

