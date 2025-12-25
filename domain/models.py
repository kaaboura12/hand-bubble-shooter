"""
Domain models for hand detection
"""
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum


class HandLandmark(Enum):
    """MediaPipe hand landmarks"""
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


@dataclass
class Point:
    """Represents a 2D point"""
    x: float
    y: float
    z: Optional[float] = None


@dataclass
class Hand:
    """Represents a detected hand with landmarks"""
    landmarks: List[Point]
    handedness: str  # "Left" or "Right"
    confidence: float


@dataclass
class DetectionResult:
    """Result of hand detection"""
    hands: List[Hand]
    timestamp: float


@dataclass
class Bubble:
    """Represents a floating bubble"""
    x: float  # Normalized x position (0.0 to 1.0)
    y: float  # Normalized y position (0.0 to 1.0)
    radius: int  # Radius in pixels
    velocity_x: float  # Horizontal velocity
    velocity_y: float  # Vertical velocity
    color: tuple  # BGR color tuple
    id: int  # Unique identifier
    points: int = 10  # Points when popped

