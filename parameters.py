import numpy as np

GRAVITATION_ACCELERATION_ms2 = 9.81 # Acceleration due to gravity [m/s^2]

# --- Atmospheric Parameters (for baro) ---
PRESSURE_SEA_LEVEL_Pa = 101325 # Sea level standard atmospheric pressure, Pa
ROOM_TEMPERATURE_K = 295.15 # Standard room temperature, K (22 deg C)
TEMP_LAPSE_RATE_Km = 0.00182 # Temperature lapse rate [K/n]
UNIVERSAL_GAS_CONSTANT_JmolK = 8.31447 # Universal gas constant [J/(mol*K)]
AIR_MOLAR_MASS_kgmol = 0.0289644 # Molar mass of dry air [kg/mol]

# --- Neural Net Parameters ---
TRAINING_FILES = [
    "log_0_2025-8-15-20-14-26", 
    "log_2_2025-8-15-12-31-04", 
    "log_0_2025-8-15-13-19-30",
    "04_48_00",
    "04_50_46",
    "04_53_08",
    "05_11_56",
    "05_14_55",
    "05_17_12",
]
TEST_FILES = [

    "05_36_10",
    "05_38_54",
    "05_41_13",
]

NUM_EPOCHS = 200 # Numbers of times to loop over data
NUM_NEURONS = 12 # Number of neurons per hidden layer
NUM_LAYERS = 2 # Number of hidden layers between the input and output
LEARNING_RATE = 1e-3
BATCH_SIZE = 32

# --- Low Pass Filters ---
ACCEL_LPF_ALPHA_X = 0.05
ACCEL_LPF_ALPHA_Y = 0.05
ACCEL_LPF_ALPHA_Z = 0.05
MAG_LPF_ALPHA = 0.5
GYRO_LPF_ALPHA = 0.1
BARO_LPF_ALPHA = 0.4

# EKF Low Pass Filter 
ORIENTATION_LPF_ALPHA = np.array([
    1, 1, 1, 1, # Quaternion (4)
    0.1, 0.1, 0.1,  # Gyro bias (3)
    0.1, 0.1, 0.1,  # Mag bias (3)
])

POSITION_LPF_ALPHA = np.array([
    1, 1, 1, # Position (3)
    1, 1, 0.9, # Velocity (3)
])

# EKF Testing Parameters
INIT_TIME = 2.0 # seconds
