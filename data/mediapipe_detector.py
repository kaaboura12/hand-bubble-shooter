"""
MediaPipe Hands implementation of IHandDetector
"""
import time
import numpy as np
from typing import Optional
import mediapipe as mp
from domain.interfaces import IHandDetector
from domain.models import DetectionResult, Hand, Point


class MediaPipeHandDetector(IHandDetector):
    """MediaPipe Hands detector implementation"""
    
    def __init__(self, 
                 static_image_mode: bool = False,
                 max_num_hands: int = 2,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize MediaPipe Hand Detector
        
        Args:
            static_image_mode: If True, treat input images as static
            max_num_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for hand tracking
        """
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        self.mp_hands = mp.solutions.hands
        self.hands = None
        self._initialized = False
    
    def initialize(self) -> bool:
        """Initialize MediaPipe Hands"""
        try:
            self.hands = self.mp_hands.Hands(
                static_image_mode=self.static_image_mode,
                max_num_hands=self.max_num_hands,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )
            self._initialized = True
            return True
        except Exception as e:
            print(f"Failed to initialize MediaPipe Hands: {e}")
            self._initialized = False
            return False
    
    def detect(self, image: np.ndarray) -> Optional[DetectionResult]:
        """
        Detect hands in image
        
        Args:
            image: Input image in BGR format
            
        Returns:
            DetectionResult or None
        """
        if not self._initialized or self.hands is None:
            return None
        
        # Convert BGR to RGB
        rgb_image = np.flip(image, axis=-1)
        
        # Process the image
        results = self.hands.process(rgb_image)
        
        if not results.multi_hand_landmarks:
            return None
        
        hands = []
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Get handedness
            handedness = results.multi_handedness[idx].classification[0].label
            
            # Extract landmarks
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.append(Point(
                    x=landmark.x,
                    y=landmark.y,
                    z=landmark.z
                ))
            
            # Get confidence
            confidence = results.multi_handedness[idx].classification[0].score
            
            hands.append(Hand(
                landmarks=landmarks,
                handedness=handedness,
                confidence=confidence
            ))
        
        return DetectionResult(
            hands=hands,
            timestamp=time.time()
        )
    
    def release(self) -> None:
        """Release MediaPipe resources"""
        if self.hands:
            self.hands.close()
            self.hands = None
        self._initialized = False
