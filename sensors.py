import numpy as np
import signal_filters
from collections import deque

class Sensor:
    def __init__(self, name, sensor_topic, keys, alpha=None, enable_extrapolation=True):
        """
        Dimension-agnostic sensor class with filtering and extrapolation.
        
        Parameters:
        name : str
            Sensor name for identification
        sensor_topic : ULog topic
            Topic containing the sensor data
        keys : list[str]
            List of keys to extract from the topic
        alpha : np.ndarray
            Filter coefficient(s). If array, must match dimension of keys
        enable_extrapolation : bool
            Whether to extrapolate data (default: True)
        """
        self.name = name
        self.topic = sensor_topic
        self.keys = keys
        self.filter = signal_filters.LowPassFilter(alpha) if alpha is not None else None
        self.enable_extrapolation = enable_extrapolation

        self.time = self.topic.data["timestamp"] * 1e-6
        self.idx = 0
        self.current_data = np.array([
            self.topic.data[k][self.idx] for k in self.keys
        ])
        
        # Keep last 5 finite differences
        self.finite_diffs = deque(maxlen=5)
        self.finite_diffs.append(np.zeros(len(keys)))
        self.dt = self.time[1] - self.time[0]

    def update_data(self, current_time_s):
        """
        Update sensor data to current time.
        Returns: bool indicating if more data is available
        """
        while self.idx + 1 < len(self.time) and self.time[self.idx + 1] <= current_time_s:
            self.idx += 1
            previous_data = self.current_data.copy()
            
            new_data = np.array([
                self.topic.data[k][self.idx] for k in self.keys
            ])
            
            if self.filter is not None:
                self.current_data = self.filter.update(new_data)
            else:
                self.current_data = new_data
                
            self.finite_diffs.append((self.current_data - previous_data) / self.dt)
        
        return self.idx < len(self.time) - 1

    def get_data(self, current_time_s):
        """
        Get sensor data at requested time, optionally with extrapolation.
        
        Parameters:
        current_time_s : float
            Current time in seconds
            
        Returns:
        np.ndarray: Sensor data vector
        """
        self.update_data(current_time_s)

        if not self.enable_extrapolation: return self.current_data

        avg_slope = np.mean(self.finite_diffs, axis=0)
        time_to_extrapolate = current_time_s - self.time[self.idx]
        return self.current_data + avg_slope * time_to_extrapolate
        
