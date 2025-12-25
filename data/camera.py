"""
OpenCV camera implementation
"""
import cv2
import numpy as np
from typing import Optional
from domain.interfaces import ICamera


class OpenCVCamera(ICamera):
    """OpenCV camera implementation"""
    
    def __init__(self):
        self.cap = None
        self._opened = False
    
    def open(self, camera_index: int = 0) -> bool:
        """Open camera"""
        try:
            self.cap = cv2.VideoCapture(camera_index)
            if self.cap.isOpened():
                # Set camera properties for better performance
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self._opened = True
                return True
            else:
                self._opened = False
                return False
        except Exception as e:
            print(f"Failed to open camera: {e}")
            self._opened = False
            return False
    
    def read(self) -> Optional[tuple]:
        """Read frame from camera"""
        if not self._opened or self.cap is None:
            return None
        
        ret, frame = self.cap.read()
        if ret:
            return (True, frame)
        return None
    
    def release(self) -> None:
        """Release camera resources"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self._opened = False
    
    def is_opened(self) -> bool:
        """Check if camera is opened"""
        return self._opened and self.cap is not None and self.cap.isOpened()

