from pyulog import ULog
import numpy as np
import torch
import copy
from scipy.spatial.transform import Rotation as R
from estimator import Orientation_EKF, Position_Velocity_EKF
from sensors import Sensor
from neural_net import Residual_Estimator
from parameters import (
    TRAINING_FILES, TEST_FILES,
    ACCEL_LPF_ALPHA_X, ACCEL_LPF_ALPHA_Y, ACCEL_LPF_ALPHA_Z,
    MAG_LPF_ALPHA, GYRO_LPF_ALPHA, BARO_LPF_ALPHA,
    NUM_LAYERS, NUM_NEURONS,
    INIT_TIME,
)

# ---- Constants ----
PRED_DT = 0.004
UPDATE_DT = 0.004
PRED_STEPS_PER_UPDATE = int(UPDATE_DT / PRED_DT)

# ---- Helpers ----
def acc_mag_to_quaternion(acc, mag):
    """
    Compute orientation quaternion from accelerometer and magnetometer vectors.
    This follows a standard method:
    - Normalize acc and mag
    - Compute pitch and roll from acc
    - Compensate mag with pitch and roll to compute yaw
    - Form quaternion from roll, pitch, yaw
    """
    # Normalize accelerometer measurement (gravity vector)
    acc = - acc / np.linalg.norm(acc)
    # Calculate pitch and roll
    pitch = np.arcsin(acc[0])
    roll = np.arctan2(acc[1], acc[2])

    # Normalize magnetometer measurement
    mag = mag / np.linalg.norm(mag)
    # Compensate magnetometer readings with pitch and roll
    mx = mag[0]*np.cos(pitch)*np.cos(roll) + mag[1]*np.cos(pitch)*np.sin(roll) - mag[2]*np.sin(pitch)
    mz = mag[0]*np.cos(roll)*np.sin(pitch) + mag[1]*np.sin(pitch)*np.sin(roll) + mag[2]*np.cos(pitch)
    # Compute yaw
    yaw = np.arctan2(mz, mx)

    # Convert roll, pitch, yaw to quaternion
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return np.array([qw, qx, qy, qz])

# ---- Main Testing Function ----
def run_test(filename, with_nn=False):
    # ---- Load ULog ----
    ulog = ULog(f'flight_data\{filename}.ulg')

    # ---- Sensor topics ----
    mag_msgs   = next(m for m in ulog.data_list if m.name == 'sensor_mag')
    sensor_combined_msgs = next(s for s in ulog.data_list if s.name == 'sensor_combined')
    baro_msgs  = next(b for b in ulog.data_list if b.name == 'sensor_baro')

    timestamps_mag = mag_msgs.data['timestamp'] * 1e-6
    timestamps_combined = sensor_combined_msgs.data['timestamp'] * 1e-6
    timestamps_baro = baro_msgs.data['timestamp'] * 1e-6

    # ---- Sensor Instances ----
    sensor_mag = Sensor('magnetometer', mag_msgs, ['x', 'y', 'z'], alpha=np.ones(3)*MAG_LPF_ALPHA)
    sensor_combined_accel = Sensor(
        'accelerometer', sensor_combined_msgs,
        ['accelerometer_m_s2[0]', 'accelerometer_m_s2[1]', 'accelerometer_m_s2[2]'],
        alpha=np.array([ACCEL_LPF_ALPHA_X, ACCEL_LPF_ALPHA_Y, ACCEL_LPF_ALPHA_Z])
    )
    sensor_combined_gyro = Sensor(
        'gyroscope', sensor_combined_msgs,
        ['gyro_rad[0]', 'gyro_rad[1]', 'gyro_rad[2]'],
        alpha=np.ones(3)*GYRO_LPF_ALPHA
    )
    sensor_baro = Sensor(
        'barometer', baro_msgs, ['pressure'],
        alpha=np.ones(1)*BARO_LPF_ALPHA, enable_extrapolation=False
    )

    # ---- PX4 Estimator Attitude ----
    att_topic = next(x for x in ulog.data_list if x.name == 'vehicle_attitude')

    # ---- Time Alignment ----
    end_time = min(
        timestamps_mag[-1],
        timestamps_combined[-1],
        timestamps_baro[-1]
    )
    time_steps = np.arange(INIT_TIME, end_time, UPDATE_DT)

    print("EKF boot up")

    # ---- Initialize EKF ----

    init_accel = sensor_combined_accel.get_data(INIT_TIME)
    init_mag = sensor_mag.get_data(INIT_TIME)
    init_gyro = sensor_combined_gyro.get_data(INIT_TIME)
    init_baro = sensor_baro.get_data(INIT_TIME)

    init_quat = acc_mag_to_quaternion(init_accel, init_mag)
    euler = R.from_quat(
        [init_quat[1], init_quat[2], init_quat[3], init_quat[0]]
    ).as_euler('xyz', degrees=True) # scipy uses x,y,z,w
    print("Initial EKF Euler angles:", euler)

    def init_orientation_ekf():
        return Orientation_EKF(
            gyroscope_reading=init_gyro,
            accelerometer_reading=init_accel,
            magnetometer_reading=init_mag,
            prediction_timestep=PRED_DT,
            initial_quaternion=init_quat
        )

    def init_position_ekf():
        ekf = Position_Velocity_EKF(
            accelerometer_reading=init_accel,
            barometer_reading=init_baro,
            prediction_timestep=PRED_DT,
            initial_quaternion=init_quat
        )
        return ekf

    # Create EKF instances for each scenario
    orientation_full = init_orientation_ekf()
    position_full    = init_position_ekf()

    orientation_pred = copy.deepcopy(orientation_full)
    position_pred    = copy.deepcopy(position_full)

    orientation_upd  = copy.deepcopy(orientation_full)
    position_upd     = copy.deepcopy(position_full)

    if with_nn:
        orientation_full_nn = copy.deepcopy(orientation_full)
        position_full_nn    = copy.deepcopy(position_full)

    print("EKF initialized")

    # ---- Scenario 1: Full Predict + Update ----
    for curr_time in time_steps:
        # Input sensor readings into EKFs
        accelerometer_filtered = sensor_combined_accel.get_data(curr_time)
        gyro_filtered = sensor_combined_gyro.get_data(curr_time)
        magnetometer_filtered = sensor_mag.get_data(curr_time)
        baro_pressure_filtered = sensor_baro.get_data(curr_time)
    
        orientation_full.set_observation_and_input(
            np.concatenate([accelerometer_filtered, magnetometer_filtered]),
            gyro_filtered
        )
        position_full.set_observation_and_input(
            baro_pressure_filtered,
            accelerometer_filtered,
        )
        # Prediction steps
        orientation_full.predict()
        position_full.update_orientation(orientation_full.current_state[:4])
        position_full.predict()

        # Update Orientation EKF
        orientation_full.update()
        orientation_full.update_ekf_history(time=curr_time)

        # Update Position EKF
        position_full.update_orientation(orientation_full.current_state[:4])
        position_full.update()
        position_full.update_ekf_history(time=curr_time)

    # ---- Scenario 2: Predict  ----
    # Sensor Restart
    sensor_mag = Sensor('magnetometer', mag_msgs, ['x', 'y', 'z'], alpha=np.ones(3)*MAG_LPF_ALPHA)
    sensor_combined_accel = Sensor(
        'accelerometer', sensor_combined_msgs,
        ['accelerometer_m_s2[0]', 'accelerometer_m_s2[1]', 'accelerometer_m_s2[2]'],
        alpha=np.array([ACCEL_LPF_ALPHA_X, ACCEL_LPF_ALPHA_Y, ACCEL_LPF_ALPHA_Z])
    )
    sensor_combined_gyro = Sensor(
        'gyroscope', sensor_combined_msgs,
        ['gyro_rad[0]', 'gyro_rad[1]', 'gyro_rad[2]'],
        alpha=np.ones(3)*GYRO_LPF_ALPHA
    )
    sensor_baro = Sensor(
        'barometer', baro_msgs, ['pressure'],
        alpha=np.ones(1)*BARO_LPF_ALPHA, enable_extrapolation=False
    )

    start_time = max(
        timestamps_mag[0],
        timestamps_combined[0],
        timestamps_baro[0]
    )
    for curr_time in np.arange(start_time, end_time, PRED_DT):
        accelerometer_filtered = sensor_combined_accel.get_data(curr_time)
        gyro_filtered = sensor_combined_gyro.get_data(curr_time)
        orientation_pred.current_input = np.array(gyro_filtered)
        position_pred.current_input = np.array(accelerometer_filtered)
        position_pred.update_orientation(orientation_pred.current_state[:4])

        orientation_pred.predict()
        position_pred.predict()
        orientation_pred.update_ekf_history(time=curr_time)
        position_pred.update_ekf_history(time=curr_time)

    # ---- Scenario 3: Update Only ----
    # Sensor Restart
    sensor_mag = Sensor('magnetometer', mag_msgs, ['x', 'y', 'z'], alpha=np.ones(3)*MAG_LPF_ALPHA)
    sensor_combined_accel = Sensor(
        'accelerometer', sensor_combined_msgs,
        ['accelerometer_m_s2[0]', 'accelerometer_m_s2[1]', 'accelerometer_m_s2[2]'],
        alpha=np.array([ACCEL_LPF_ALPHA_X, ACCEL_LPF_ALPHA_Y, ACCEL_LPF_ALPHA_Z])
    )
    sensor_combined_gyro = Sensor(
        'gyroscope', sensor_combined_msgs,
        ['gyro_rad[0]', 'gyro_rad[1]', 'gyro_rad[2]'],
        alpha=np.ones(3)*GYRO_LPF_ALPHA
    )
    sensor_baro = Sensor(
        'barometer', baro_msgs, ['pressure'],
        alpha=np.ones(1)*BARO_LPF_ALPHA, enable_extrapolation=False
    )

    for curr_time in time_steps:
        accelerometer_filtered = sensor_combined_accel.get_data(curr_time)
        gyro_filtered = sensor_combined_gyro.get_data(curr_time)
        magnetometer_filtered = sensor_mag.get_data(curr_time)
        baro_pressure_filtered = sensor_baro.get_data(curr_time)

        # Orientation Update - manual
        quaternion = np.array([acc_mag_to_quaternion(
            np.array(accelerometer_filtered),
            np.array(magnetometer_filtered)
        )])
        orientation_upd.current_state[:4] = quaternion
        orientation_upd.update_ekf_history(time=curr_time)

        # Position EKF Update — manual
        pz = (
            np.log(baro_pressure_filtered[0] / position_upd.PRESSURE_SEA_LEVEL_Pa) *
            position_upd.ROOM_TEMPERATURE_K *
            position_upd.UNIVERSAL_GAS_CONSTANT_JmolK / (
                position_upd.AIR_MOLAR_MASS_kgmol *
                position_upd.GRAVITATION_ACCELERATION_ms2
            )
        ) # BAROMETRIC FORMULA
        position_upd.current_state[2] = pz
        position_upd.update_ekf_history(time=curr_time)

    # ---- Scenario 4: Full EKF + NN ----
    if with_nn:
        # Sensor Restart
        sensor_mag = Sensor('magnetometer', mag_msgs, ['x', 'y', 'z'], alpha=np.ones(3)*MAG_LPF_ALPHA)
        sensor_combined_accel = Sensor(
            'accelerometer', sensor_combined_msgs,
            ['accelerometer_m_s2[0]', 'accelerometer_m_s2[1]', 'accelerometer_m_s2[2]'],
            alpha=np.array([ACCEL_LPF_ALPHA_X, ACCEL_LPF_ALPHA_Y, ACCEL_LPF_ALPHA_Z])
        )
        sensor_combined_gyro = Sensor(
            'gyroscope', sensor_combined_msgs,
            ['gyro_rad[0]', 'gyro_rad[1]', 'gyro_rad[2]'],
            alpha=np.ones(3)*GYRO_LPF_ALPHA
        )
        sensor_baro = Sensor(
            'barometer', baro_msgs, ['pressure'],
            alpha=np.ones(1)*BARO_LPF_ALPHA, enable_extrapolation=False
        )

        # Load Neural Net model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Instantiate model with correct sizes (adjust these to your training config)
        num_inputs = 20     # EKF state + sensor features
        num_outputs = 10    # residual output dimension
        num_units = NUM_NEURONS
        num_layers = NUM_LAYERS

        model = Residual_Estimator(num_inputs, num_outputs, num_units, num_layers)
        model.load_state_dict(torch.load(
            'neural_net/residual_estimator_model.pth', map_location=device, weights_only=False
        ))
        model.to(device)
        model.eval()

        normalization_stats = np.load('neural_net/normalization_stats.npz')
        target_mean = normalization_stats['target_mean']
        target_std = normalization_stats['target_std']
        input_mean = normalization_stats["input_mean"]
        input_std = normalization_stats["input_std"]

        # Testing loop
        for curr_time in time_steps:
            # Prediction steps
            for _ in range(PRED_STEPS_PER_UPDATE):
                curr_time += PRED_DT
                orientation_full_nn.predict()
                position_full_nn.update_orientation(orientation_full_nn.current_state[:4])
                position_full_nn.predict()
                orientation_full_nn.update_ekf_history(time=curr_time)
                position_full_nn.update_ekf_history(time=curr_time)

            accelerometer_filtered = sensor_combined_accel.get_data(curr_time)
            gyro_filtered = sensor_combined_gyro.get_data(curr_time)
            magnetometer_filtered = sensor_mag.get_data(curr_time)
            baro_pressure_filtered = sensor_baro.get_data(curr_time)

            # Neural Net Correction
            ekf_state = np.concat((
                position_full_nn.current_state, # pos, vel
                orientation_full_nn.current_state[:4] # quaternion
            ))
            nn_input_np = np.concat((
                ekf_state,
                accelerometer_filtered,
                gyro_filtered,
                baro_pressure_filtered,
                magnetometer_filtered
            ))

            nn_input_norm = (nn_input_np - input_mean) / input_std
            nn_input_tensor = torch.tensor(nn_input_norm, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                predicted_residual = model(nn_input_tensor)  # shape (1, num_outputs)
            predicted_residual_np = predicted_residual.cpu().numpy().flatten()

            predicted_residual_np_unnormalized = predicted_residual_np * target_std + target_mean

            # Apply residual correction to EKF estimate
            corrected_state = ekf_state + predicted_residual_np_unnormalized

            position_full_nn.update_state(corrected_state[:6])
            orientation_full_nn.update_state(np.concat((
                corrected_state[6:],
                orientation_full_nn.current_state[4:]
            )))
            
            # Update Orientation EKF
            orientation_full_nn.set_observation(
                np.array([
                    accelerometer_filtered[0],
                    accelerometer_filtered[1],
                    accelerometer_filtered[2],
                    magnetometer_filtered[0], magnetometer_filtered[1], magnetometer_filtered[2],
                ]),
                gyro_filtered
            )
            orientation_full_nn.update()
            orientation_full_nn.update_ekf_history(time=curr_time)

            # Update Position EKF only when new barometer data is available
            position_full_nn.set_observation(
                baro_pressure_filtered,
                accelerometer_filtered,
            )
            position_full_nn.update_orientation(orientation_full_nn.current_state[:4])
            position_full_nn.update()
            position_full_nn.update_ekf_history(time=curr_time)


    # ---- Export Results ----
    orientation_full.export(f'EKF_data/{filename}_orientation_ekf_full')
    position_full.export(f'EKF_data/{filename}_position_ekf_full')

    orientation_pred.export(f'EKF_data/{filename}_orientation_ekf_pred_only')
    position_pred.export(f'EKF_data/{filename}_position_ekf_pred_only')

    orientation_upd.export(f'EKF_data/{filename}_orientation_ekf_update_only')
    position_upd.export(f'EKF_data/{filename}_position_ekf_update_only')

    if with_nn:
        orientation_full_nn.export(f'EKF_data/{filename}_orientation_ekf_full_nn')
        position_full_nn.export(f'EKF_data/{filename}_position_ekf_full_nn')

    print("Test EKF scenarios completed and CSVs exported")


if __name__=="__main__":
    for file in TRAINING_FILES:
        run_test(file)
    for file in TEST_FILES:
        run_test(file)
