"""
Action smoothing module for robot navigation.

This module provides the ActionSmoother class to reduce oscillation in robot movement
by tracking action history and applying momentum to action selection.
"""

from collections import deque
import numpy as np


class ActionSmoother:
    """
    Smooths action selection to reduce oscillation in robot movement.

    The ActionSmoother tracks recent actions and applies momentum to prefer
    actions similar to those recently taken, while also detecting thrashing
    (rapid alternation between opposite actions).

    Attributes:
        history_length (int): Number of recent actions to track
        momentum_weight (float): Weight given to momentum vs desired action (0-1)
        action_history (deque): Deque storing recent actions
    """

    def __init__(self, history_length=5, momentum_weight=0.4):
        """
        Initialize the ActionSmoother.

        Args:
            history_length (int): Number of recent actions to track in history.
                Default is 5.
            momentum_weight (float): Weight for momentum in action selection.
                Range [0, 1] where 0 = ignore momentum, 1 = only momentum.
                Default is 0.4 (40% momentum, 60% desired action).
        """
        self.history_length = history_length
        self.momentum_weight = momentum_weight
        self.action_history = deque(maxlen=history_length)

    def add_action(self, action):
        """
        Record an action that was taken.

        Args:
            action (int): The action that was executed (-5 to +5)
        """
        self.action_history.append(action)

    def get_smoothed_action(self, available_actions, desired_action):
        """
        Select action considering momentum from recent actions.

        Balances between the desired action (steering toward goal) and
        momentum from recent action history. If there's insufficient history,
        defaults to selecting the action closest to desired.

        Args:
            available_actions (list): List of safe actions (collision prob < threshold)
            desired_action (int): The action that would steer toward the goal

        Returns:
            int: Best action balancing desire and momentum
        """
        if not available_actions:
            return 0  # Default to straight if no actions available

        # If we don't have enough history, just return closest to desired
        if len(self.action_history) < 2:
            min_diff = float('inf')
            closest_action = available_actions[0]
            for action in available_actions:
                diff = abs(desired_action - action)
                if diff < min_diff:
                    min_diff = diff
                    closest_action = action
            return closest_action

        # Calculate momentum (average of recent actions)
        momentum_action = np.mean(list(self.action_history))

        # Score each available action based on both desired action and momentum
        best_action = available_actions[0]
        best_score = float('inf')

        for action in available_actions:
            # Distance from desired action (what we want to do)
            desire_distance = abs(desired_action - action)

            # Distance from momentum (what we've been doing)
            momentum_distance = abs(momentum_action - action)

            # Weighted combination: lower is better
            # momentum_weight controls the balance between momentum and desire
            score = (self.momentum_weight * momentum_distance +
                    (1 - self.momentum_weight) * desire_distance)

            if score < best_score:
                best_score = score
                best_action = action

        return best_action

    def detect_thrashing(self, threshold=3):
        """
        Detect if robot is rapidly switching between opposite actions.

        Thrashing is detected by counting sign changes in the action history.
        A sign change occurs when consecutive actions have opposite signs
        (e.g., turning left then right, or right then left).

        Args:
            threshold (int): Number of sign changes to consider as thrashing.
                Default is 3.

        Returns:
            bool: True if thrashing detected, False otherwise
        """
        if len(self.action_history) < 3:
            return False

        # Count sign changes in the action history
        sign_changes = 0
        history_list = list(self.action_history)

        for i in range(1, len(history_list)):
            prev_action = history_list[i - 1]
            curr_action = history_list[i]

            # Skip if either action is zero (going straight)
            if prev_action == 0 or curr_action == 0:
                continue

            # Check if signs are different (one positive, one negative)
            if (prev_action > 0 and curr_action < 0) or (prev_action < 0 and curr_action > 0):
                sign_changes += 1

        return sign_changes >= threshold

    def reset(self):
        """
        Clear action history.

        Should be called when the robot collides or reaches a goal,
        as the action history is no longer relevant for the new situation.
        """
        self.action_history.clear()
