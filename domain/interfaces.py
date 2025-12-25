"""
Domain interfaces - Abstract contracts for hand detection
"""
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class IHandDetector(ABC):
    """Interface for hand detection service"""
    
    @abstractmethod
    def detect(self, image: np.ndarray) -> Optional['DetectionResult']:
        """
        Detect hands in an image
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            DetectionResult containing detected hands, or None if no hands detected
        """
        pass
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the hand detector
        
        Returns:
            True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def release(self) -> None:
        """Release resources"""
        pass


class ICamera(ABC):
    """Interface for camera service"""
    
    @abstractmethod
    def open(self, camera_index: int = 0) -> bool:
        """
        Open camera
        
        Args:
            camera_index: Index of the camera to use
            
        Returns:
            True if camera opened successfully
        """
        pass
    
    @abstractmethod
    def read(self) -> Optional[tuple]:
        """
        Read frame from camera
        
        Returns:
            Tuple of (success: bool, frame: np.ndarray) or None if failed
        """
        pass
    
    @abstractmethod
    def release(self) -> None:
        """Release camera resources"""
        pass
    
    @abstractmethod
    def is_opened(self) -> bool:
        """Check if camera is opened"""
        pass

