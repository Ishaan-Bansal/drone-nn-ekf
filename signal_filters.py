import numpy as np

class LowPassFilter:
    def __init__(self, alpha, initial_array=None):
        """
        Simple exponential moving average (EMA) low-pass filter.

        Parameters:
        alpha : dimension x np.ndarray(float)
            Smoothing factor, between 0 and 1. Smaller means more smoothing.

        initial_array : dimension x float or np.ndarray, optional
            Initial filtered value.
            raise ValueError("Alpha must be in (0, 1] for all dimensions.")
        """
        
        self.alpha = alpha
        self.filtered_array = (
            np.array(initial_array, dtype=float) if initial_array is not None
            else np.zeros_like(alpha, dtype=float)
        )
        self.initialized = False

    def update(self, new_array):
        """
        Process one new data point and return the filtered result.
        
        Parameters:
        new_array : dimension x np.ndarray
            The new incoming sample.

        Returns:
        float : The updated filtered value.
        """
        if not self.initialized:
            # First datapoint initializes the filter
            self.filtered_array = new_array
            self.initialized = True
        else:
            self.filtered_array = (
                self.alpha * new_array +
                (1 - self.alpha) * self.filtered_array
            )
        return self.filtered_array
    
    def reset(self):
        self.initialized = False

class ZScoreFilter_1D:
    def __init__(self, threshold=3.0):
        self.threshold = threshold
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0  # Sum of squares of differences from the current mean

    def update(self, new_value):
        self.n += 1
        delta = new_value - self.mean
        self.mean += delta / self.n
        delta2 = new_value - self.mean
        self.M2 += delta * delta2

        # For the first value, variance is undefined, so just return new_value
        if self.n < 2:
            return new_value

        variance = self.M2 / (self.n - 1)
        std = np.sqrt(variance) if variance > 0 else 1.0
        z_score = (new_value - self.mean) / std

        # If outlier, return mean instead (could use median or previous value)
        if abs(z_score) > self.threshold:
            return self.mean  # Or use median, previous value, or np.nan
        else:
            return new_value

class ZScoreFilter_3D:
    def __init__(self, threshold=3.0):
        self.x = ZScoreFilter_1D(threshold)
        self.y = ZScoreFilter_1D(threshold)
        self.z = ZScoreFilter_1D(threshold)

    def update(self, new_array):
        self.x.update(new_array[0])
        self.y.update(new_array[1])
        self.z.update(new_array[2])

