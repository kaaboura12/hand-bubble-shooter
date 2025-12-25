"""
Bubble game logic and management
"""
import random
import time
import numpy as np
from typing import List, Optional
from domain.models import Bubble, Hand, Point, HandLandmark


class BubbleGame:
    """Manages bubble game state and logic"""
    
    def __init__(self, 
                 screen_width: int,
                 screen_height: int,
                 max_bubbles: int = 5,
                 spawn_rate: float = 0.8,
                 min_radius: int = 30,
                 max_radius: int = 50):
        """
        Initialize bubble game
        
        Args:
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
            max_bubbles: Maximum number of bubbles on screen
            spawn_rate: Bubbles per second
            min_radius: Minimum bubble radius
            max_radius: Maximum bubble radius
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.max_bubbles = max_bubbles
        self.spawn_rate = spawn_rate
        self.min_radius = min_radius
        self.max_radius = max_radius
        
        self.bubbles: List[Bubble] = []
        self.next_bubble_id = 0
        self.last_spawn_time = time.time()
        self.score = 0
        self.bubbles_popped = 0
        
        # Bubble colors (BGR format for OpenCV)
        self.bubble_colors = [
            (255, 100, 100),  # Light blue
            (100, 255, 100),  # Light green
            (100, 100, 255),  # Light red
            (255, 255, 100),  # Cyan
            (255, 100, 255),  # Magenta
            (100, 255, 255),  # Yellow
        ]
    
    def update(self, delta_time: float) -> None:
        """
        Update game state
        
        Args:
            delta_time: Time since last update in seconds
        """
        current_time = time.time()
        
        # Spawn new bubbles
        if (len(self.bubbles) < self.max_bubbles and 
            current_time - self.last_spawn_time >= 1.0 / self.spawn_rate):
            self._spawn_bubble()
            self.last_spawn_time = current_time
        
        # Update bubble positions
        bubbles_to_remove = []
        for bubble in self.bubbles:
            # Update position with frame-rate independent movement
            # Clamp delta_time to prevent large jumps
            clamped_delta = min(delta_time, 0.1)  # Max 0.1s per frame
            bubble.x += bubble.velocity_x * clamped_delta
            bubble.y += bubble.velocity_y * clamped_delta
            
            # Bounce off walls
            pixel_x = bubble.x * self.screen_width
            pixel_y = bubble.y * self.screen_height
            
            if pixel_x - bubble.radius <= 0 or pixel_x + bubble.radius >= self.screen_width:
                bubble.velocity_x *= -1
                bubble.x = max(bubble.radius / self.screen_width, 
                              min(bubble.x, 1.0 - bubble.radius / self.screen_width))
            
            if pixel_y - bubble.radius <= 0 or pixel_y + bubble.radius >= self.screen_height:
                bubble.velocity_y *= -1
                bubble.y = max(bubble.radius / self.screen_height,
                             min(bubble.y, 1.0 - bubble.radius / self.screen_height))
            
            # Remove bubbles that are too old or off screen (safety check)
            if bubble.x < -0.1 or bubble.x > 1.1 or bubble.y < -0.1 or bubble.y > 1.1:
                bubbles_to_remove.append(bubble)
        
        # Remove bubbles
        for bubble in bubbles_to_remove:
            self.bubbles.remove(bubble)
    
    def _spawn_bubble(self) -> None:
        """Spawn a new bubble"""
        radius = random.randint(self.min_radius, self.max_radius)
        
        # Spawn from random edge with slower, more controlled velocities
        side = random.randint(0, 3)
        if side == 0:  # Top
            x = random.uniform(0.15, 0.85)
            y = 0.0
            velocity_x = random.uniform(-0.15, 0.15)
            velocity_y = random.uniform(0.08, 0.25)
        elif side == 1:  # Right
            x = 1.0
            y = random.uniform(0.15, 0.85)
            velocity_x = random.uniform(-0.25, -0.08)
            velocity_y = random.uniform(-0.15, 0.15)
        elif side == 2:  # Bottom
            x = random.uniform(0.15, 0.85)
            y = 1.0
            velocity_x = random.uniform(-0.15, 0.15)
            velocity_y = random.uniform(-0.25, -0.08)
        else:  # Left
            x = 0.0
            y = random.uniform(0.15, 0.85)
            velocity_x = random.uniform(0.08, 0.25)
            velocity_y = random.uniform(-0.15, 0.15)
        
        bubble = Bubble(
            x=x,
            y=y,
            radius=radius,
            velocity_x=velocity_x,
            velocity_y=velocity_y,
            color=random.choice(self.bubble_colors),
            id=self.next_bubble_id,
            points=radius // 5  # More points for bigger bubbles
        )
        
        self.next_bubble_id += 1
        self.bubbles.append(bubble)
    
    def check_collisions(self, pointer_x: float, pointer_y: float, is_shooting: bool) -> List[Bubble]:
        """
        Check for collisions between pointer and bubbles when shooting
        
        Args:
            pointer_x: Pointer x position in normalized coordinates (0.0 to 1.0)
            pointer_y: Pointer y position in normalized coordinates (0.0 to 1.0)
            is_shooting: True if hand is closed (shooting), False otherwise
            
        Returns:
            List of bubbles that were hit
        """
        hit_bubbles = []
        
        if not is_shooting:
            return hit_bubbles
        
        # Convert normalized coordinates to pixel coordinates
        pointer_pixel_x = pointer_x * self.screen_width
        pointer_pixel_y = pointer_y * self.screen_height
        
        # Check collision with each bubble
        for bubble in self.bubbles:
            if bubble in hit_bubbles:
                continue
            
            bubble_x = bubble.x * self.screen_width
            bubble_y = bubble.y * self.screen_height
            
            # Calculate distance from pointer to bubble center
            distance = np.sqrt((pointer_pixel_x - bubble_x)**2 + (pointer_pixel_y - bubble_y)**2)
            
            # Check if pointer is within bubble radius (with slight tolerance for better feel)
            if distance <= bubble.radius * 1.1:  # 10% larger hitbox for better feel
                hit_bubbles.append(bubble)
                self.score += bubble.points
                self.bubbles_popped += 1
        
        # Remove hit bubbles
        for bubble in hit_bubbles:
            if bubble in self.bubbles:
                self.bubbles.remove(bubble)
        
        return hit_bubbles
    
    def reset(self) -> None:
        """Reset game state"""
        self.bubbles.clear()
        self.score = 0
        self.bubbles_popped = 0
        self.last_spawn_time = time.time()

