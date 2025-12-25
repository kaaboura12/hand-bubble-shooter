"""
Hand gesture detection utilities
"""
import numpy as np
from typing import Optional
from domain.models import Hand, HandLandmark, Point


class GestureDetector:
    """Detects hand gestures from hand landmarks"""
    
    @staticmethod
    def is_hand_closed(hand: Hand) -> bool:
        """
        Detect if hand is closed (fist)
        
        A hand is considered closed if all fingertips are close to the palm/wrist
        
        Args:
            hand: Hand with landmarks
            
        Returns:
            True if hand is closed, False otherwise
        """
        if len(hand.landmarks) < 21:
            return False
        
        # Get wrist position
        wrist = hand.landmarks[HandLandmark.WRIST.value]
        
        # Fingertip landmarks
        fingertips = [
            HandLandmark.THUMB_TIP.value,
            HandLandmark.INDEX_FINGER_TIP.value,
            HandLandmark.MIDDLE_FINGER_TIP.value,
            HandLandmark.RING_FINGER_TIP.value,
            HandLandmark.PINKY_TIP.value,
        ]
        
        # Corresponding PIP joints (for comparison)
        pip_joints = [
            HandLandmark.THUMB_IP.value,
            HandLandmark.INDEX_FINGER_PIP.value,
            HandLandmark.MIDDLE_FINGER_PIP.value,
            HandLandmark.RING_FINGER_PIP.value,
            HandLandmark.PINKY_PIP.value,
        ]
        
        closed_fingers = 0
        
        for tip_idx, pip_idx in zip(fingertips, pip_joints):
            tip = hand.landmarks[tip_idx]
            pip = hand.landmarks[pip_idx]
            
            # Check if fingertip is below PIP joint (finger is bent)
            # For thumb, check distance to wrist
            if tip_idx == HandLandmark.THUMB_TIP.value:
                # For thumb, check if it's close to wrist
                distance_to_wrist = np.sqrt(
                    (tip.x - wrist.x)**2 + (tip.y - wrist.y)**2
                )
                if distance_to_wrist < 0.15:  # Threshold for thumb
                    closed_fingers += 1
            else:
                # For other fingers, check if tip is below PIP
                if tip.y > pip.y:
                    closed_fingers += 1
        
        # Hand is closed if at least 4 out of 5 fingers are closed
        return closed_fingers >= 4
    
    @staticmethod
    def get_index_finger_tip(hand: Hand) -> Optional[Point]:
        """
        Get index finger tip position
        
        Args:
            hand: Hand with landmarks
            
        Returns:
            Point representing index finger tip, or None if not available
        """
        if len(hand.landmarks) > HandLandmark.INDEX_FINGER_TIP.value:
            return hand.landmarks[HandLandmark.INDEX_FINGER_TIP.value]
        return None

