"""
Application flow manager - handles menu -> name input -> game
"""
from typing import Optional
from data.mediapipe_detector import MediaPipeHandDetector
from data.camera import OpenCVCamera
from presentation.menu_system import MenuSystem
from presentation.bubble_game_viewer import BubbleGameViewer


class AppFlow:
    """Manages the application flow"""
    
    def __init__(self):
        """Initialize app flow"""
        self.detector = MediaPipeHandDetector(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.camera: Optional[OpenCVCamera] = None
        self.player_name: Optional[str] = None
    
    def show_menu(self) -> str:
        """Show menu and get user selection"""
        # Initialize camera once and reuse
        if self.camera is None:
            self.camera = OpenCVCamera()
            if not self.camera.open():
                return "exit"
        
        # Initialize detector once
        if not hasattr(self.detector, '_initialized') or not self.detector._initialized:
            if not self.detector.initialize():
                return "exit"
        
        menu = MenuSystem(self.detector, self.camera)
        
        # Add menu items
        def start_game_action():
            return "start_game"
        
        def exit_action():
            return "exit"
        
        # Get frame to determine menu positions
        result = self.camera.read()
        if result is None:
            return "exit"
        
        _, frame = result
        h, w = frame.shape[:2]
        
        # Position menu items
        center_x = w // 2
        start_y = h // 2 - 30
        exit_y = h // 2 + 50
        
        menu.add_menu_item("Start Game", start_game_action, (center_x, start_y))
        menu.add_menu_item("Exit", exit_action, (center_x, exit_y))
        
        # Run menu (don't release camera)
        result = menu.run()
        return result if result else "exit"
    
    def run_game(self):
        """Run the bubble game"""
        print("\nStarting game...\n")
        
        # Reuse existing camera and detector (already initialized)
        if self.camera is None:
            self.camera = OpenCVCamera()
            if not self.camera.open():
                return
        
        game_viewer = BubbleGameViewer(self.detector, self.camera)
        game_viewer.run()
    
    def run(self):
        """Run the complete application flow"""
        print("=" * 60)
        print("Hand Detection Bubble Game")
        print("=" * 60)
        
        try:
            # Step 1: Show menu
            menu_result = self.show_menu()
            if menu_result == "exit":
                print("Exiting...")
                return
            
            # Step 2: Run game
            self.run_game()
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        except Exception as e:
            print(f"\nError: {e}")
        finally:
            # Cleanup
            if self.camera:
                self.camera.release()
            self.detector.release()
            print("\nThank you for playing!")

