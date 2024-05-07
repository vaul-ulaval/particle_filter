import numpy as np

class CircularBuffer:
    def __init__(self, max_distance, dim):
        self.dim = dim
        self.max_distance = max_distance
        self.buffer = []
        self.total_distance = 0.0


    def append(self, data):
        if len(self.buffer) > 0:
            # Calculate distance from the last pose
            last_data = self.buffer[-1]
            increment_distance = np.linalg.norm(last_data[:2] - data[:2])
            self.total_distance += increment_distance

        # Append new data with its cumulative distance
        self.buffer.append(data)

        # Remove old poses to maintain the max distance constraint
        while self.total_distance > self.max_distance:
            if len(self.buffer) < 2:
                break
            # Remove the oldest pose and update the total distance
            oldest_data = self.buffer.pop(0)
            next_oldest_data = self.buffer[0]
            removed_distance = np.linalg.norm(oldest_data[:2] - next_oldest_data[:2])
            self.total_distance -= removed_distance

    def get_path(self):
        """ Return all elements in the buffer. """
        return np.array(self.buffer)

    def __len__(self):
        return len(self.buffer)