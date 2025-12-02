"""
Spatial memory system for robot navigation to prevent oscillation loops.

This module implements a grid-based spatial memory that tracks visited positions
and provides repulsion scores to help the robot avoid getting stuck in loops.
"""

import numpy as np
from collections import deque


class SpatialMemory:
    """
    Grid-based spatial memory for tracking visited positions.

    Helps prevent oscillation loops by maintaining a history of visited grid cells
    and providing repulsion scores based on visit frequency.
    """

    def __init__(self, grid_size=30, decay_rate=0.97, max_history=100):
        """
        Initialize spatial memory.

        Args:
            grid_size: Size of spatial bins in pixels (default: 30)
            decay_rate: How quickly visit counts decay, 0-1 (default: 0.97 = 3% decay per call)
            max_history: Maximum positions to track in history deque (default: 100)
        """
        self.grid_size = grid_size
        self.decay_rate = decay_rate
        self.max_history = max_history

        # Dictionary mapping grid cell tuples (grid_x, grid_y) to visit counts
        self.visit_counts = {}

        # Deque of recent positions for oscillation detection
        self.position_history = deque(maxlen=max_history)

    def _position_to_grid(self, position):
        """
        Convert continuous position to discrete grid cell.

        Args:
            position: Tuple (x, y) or Pymunk Vec2d representing position

        Returns:
            Tuple (grid_x, grid_y) representing the grid cell
        """
        # Handle both tuple and Vec2d types
        if hasattr(position, 'x') and hasattr(position, 'y'):
            x, y = position.x, position.y
        else:
            x, y = position

        # Convert to grid coordinates by integer division
        grid_x = int(x // self.grid_size)
        grid_y = int(y // self.grid_size)

        return (grid_x, grid_y)

    def add_position(self, position):
        """
        Record a visited position.

        Args:
            position: Tuple (x, y) or Pymunk Vec2d representing the position to record
        """
        # Convert to grid cell
        grid_cell = self._position_to_grid(position)

        # Increment visit count for this cell
        if grid_cell in self.visit_counts:
            self.visit_counts[grid_cell] += 1
        else:
            self.visit_counts[grid_cell] = 1

        # Add to position history (as tuple for consistency)
        if hasattr(position, 'x') and hasattr(position, 'y'):
            self.position_history.append((position.x, position.y))
        else:
            self.position_history.append(position)

    def decay_visits(self):
        """
        Gradually forget old positions by decaying visit counts.

        Removes cells with count below 0.1 to keep memory efficient.
        """
        # Decay all visit counts
        cells_to_remove = []
        for cell, count in self.visit_counts.items():
            new_count = count * self.decay_rate
            if new_count < 0.1:
                cells_to_remove.append(cell)
            else:
                self.visit_counts[cell] = new_count

        # Remove cells with negligible counts
        for cell in cells_to_remove:
            del self.visit_counts[cell]

    def get_repulsion_score(self, position):
        """
        Get repulsion score for a position.

        Higher scores indicate more frequently visited areas that should be avoided.

        Args:
            position: Tuple (x, y) or Pymunk Vec2d representing the position to check

        Returns:
            Float repulsion score (0.0 if never visited, higher for frequently visited)
        """
        grid_cell = self._position_to_grid(position)

        # Return visit count for this cell (0 if never visited)
        return self.visit_counts.get(grid_cell, 0.0)

    def detect_oscillation(self, current_position, window=15, threshold=30.0):
        """
        Detect if robot is oscillating in a small area.

        Checks if the average distance from current position to the last N positions
        is below a threshold, indicating the robot is stuck in a loop.

        Args:
            current_position: Tuple (x, y) or Pymunk Vec2d of current position
            window: Number of recent positions to check (default: 15)
            threshold: Maximum average distance to be considered oscillating (default: 30.0 pixels)

        Returns:
            bool: True if oscillation detected, False otherwise
        """
        # Need enough history to detect oscillation
        if len(self.position_history) < window:
            return False

        # Convert current position to numpy array
        if hasattr(current_position, 'x') and hasattr(current_position, 'y'):
            current_pos = np.array([current_position.x, current_position.y])
        else:
            current_pos = np.array(current_position)

        # Get last N positions
        recent_positions = list(self.position_history)[-window:]
        recent_positions_array = np.array(recent_positions)

        # Calculate distances from current position to all recent positions
        distances = np.linalg.norm(recent_positions_array - current_pos, axis=1)

        # Calculate average distance
        avg_distance = np.mean(distances)

        # Oscillation detected if average distance is below threshold
        return avg_distance < threshold

    def reset(self):
        """
        Clear all memory.

        Should be called when robot reaches a goal to start fresh.
        """
        self.visit_counts.clear()
        self.position_history.clear()
