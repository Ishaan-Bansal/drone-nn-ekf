import numpy as np

class LowPassFilter_1D:
    def __init__(self, alpha, initial_value=0.0):
        """
        Simple exponential moving average (EMA) low-pass filter.

        Parameters:
        alpha : float
            Smoothing factor, between 0 and 1. Smaller means more smoothing.
        initial_value : float, optional
            Initial filtered value (default = 0.0)
        """
        if not (0.0 < alpha <= 1.0):
            raise ValueError("Alpha must be in (0, 1].")
        
        self.alpha = alpha
        self.filtered_value = initial_value
        self.initialized = False

    def update(self, new_value):
        """
        Process one new data point and return the filtered result.
        
        Parameters:
        new_value : float
            The new incoming sample.

        Returns:
        float : The updated filtered value.
        """
        if not self.initialized:
            # First datapoint initializes the filter
            self.filtered_value = new_value
            self.initialized = True
        else:
            self.filtered_value = (
                self.alpha * new_value +
                (1 - self.alpha) * self.filtered_value
            )
        return self.filtered_value

    def reset(self):
        self.initialized = False


class LowPassFilter_3D:
    def __init__(self, alpha=None, alpha_arr=None, initial_array=[0.0, 0.0, 0.0]):
        """
        Simple exponential moving average (EMA) low-pass filter.

        Parameters:
        alpha : float
            Smoothing factor, between 0 and 1. Smaller means more smoothing.
        initial_array : 3x float, optional
            Initial filtered value (default = [0.0, 0.0, 0.0])
        """ 
        if alpha_arr is not None:
            alpha = np.array(alpha_arr, dtype=float)
            if not np.all((0.0 < alpha) & (alpha <= 1.0)):
                raise ValueError("Alpha must be in (0, 1] for all dimensions.")
        elif alpha is not None and not (0.0 < alpha <= 1.0):
            raise ValueError("Alpha must be in (0, 1].")
        
        self.alpha = alpha
        self.filtered_array = np.array(initial_array, dtype=float)
        self.initialized = False

    def update(self, new_array):
        """
        Process one new data point and return the filtered result.
        
        Parameters:
        new_value : 3x float
            The new incoming sample.

        Returns:
        float : The updated filtered value.
        """
        new_array = np.array(new_array, dtype=float)

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

