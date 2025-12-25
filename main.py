"""
Main entry point for Hand Detection Application
"""
import sys
from presentation.app_flow import AppFlow
from presentation.hand_detection_viewer import HandDetectionViewer
from data.mediapipe_detector import MediaPipeHandDetector
from data.camera import OpenCVCamera


def main():
    """Main function to run hand detection app"""
    # Check command line arguments
    mode = "full"  # Default to full flow (menu -> name -> game)
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    
    if mode == "full" or mode == "game":
        # Full flow: Menu -> Name Input -> Game
        app_flow = AppFlow()
        app_flow.run()
    else:
        # Direct mode: Skip menu and name input
        print("=" * 50)
        print("Hand Detection App - MediaPipe Hands")
        print("=" * 50)
        
        detector = MediaPipeHandDetector(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        camera = OpenCVCamera()
        viewer = HandDetectionViewer(detector, camera)
        viewer.run()


if __name__ == "__main__":
    main()

