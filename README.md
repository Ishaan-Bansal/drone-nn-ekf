# Extended Kalman Filter + Neural Network Estimator

## Overview
The goal of this project was to determine sensor fusion techniques to estimate the state of the quadcopter without the assistance of GPS. For which, the specific filtering technique chosen was the Extended Kalman Filter augmented with a Neural Network.

The Extended Kalman Filter (EKF) is broken into:
    Orientation estimation (quaternion + gyro/mag biases)
    Position and velocity estimation

The Neural Network learns to predict and correct EKF error using PX4 reference states.

## Requirements
Python 3.8+
Install dependencies:
```pip install numpy scipy pandas matplotlib torch pyulog control sympy```

## Usage
### Using existing data
Visualize the existing results through:
```python plot_comparison.py```

### Using new data
Step 1.
- Add `.ulog` to `./flight_data/`

Step 2.
- Add filenames to either `TRAINING_FILES` or `TEST_FILES` in `parameter.py`
- Run EKF Scenarios Offline
```python ekf_offline_testing.py```

Step 3.
- Train Neural Net on new data
```python neural_net.py```

Step 4.
- Visualize the results through:
```python plot_comparison.py```

### Extra
To visualize '.ulog' files:
```python plot_flight.py```

## Details
Author: Ishaan Bansal
Contact: ishaanbansal.2003@gmail.com
