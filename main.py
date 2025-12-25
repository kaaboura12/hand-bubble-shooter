"""
Main entry point for Hand Detection Application
"""
from data.mediapipe_detector import MediaPipeHandDetector
from data.camera import OpenCVCamera
from presentation.hand_detection_viewer import HandDetectionViewer


def main():
    """Main function to run hand detection app"""
    print("=" * 50)
    print("Hand Detection App - MediaPipe Hands")
    print("=" * 50)
    
    # Initialize components (Dependency Injection)
    detector = MediaPipeHandDetector(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    camera = OpenCVCamera()
    
    # Create viewer
    viewer = HandDetectionViewer(detector, camera)
    
    # Run application
    viewer.run()


if __name__ == "__main__":
    main()

