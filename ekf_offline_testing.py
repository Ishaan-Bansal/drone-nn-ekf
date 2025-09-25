from pyulog import ULog
import numpy as np
import torch
import copy
from scipy.spatial.transform import Rotation as R
from estimator import Orientation_EKF, Position_Velocity_EKF
from signal_filters import LowPassFilter_1D, LowPassFilter_3D
from neural_net import Residual_Estimator
from parameters import (
    TRAINING_FILES, TEST_FILES,
    ACCEL_LPF_ALPHA_X, ACCEL_LPF_ALPHA_Y, ACCEL_LPF_ALPHA_Z,
    MAG_LPF_ALPHA, GYRO_LPF_ALPHA, BARO_LPF_ALPHA, VEL_Z_LPF_ALPHA,
    NUM_LAYERS, NUM_NEURONS
)

# ---- Constants ----
PRED_DT = 0.004
UPDATE_DT = 0.004
PRED_STEPS_PER_UPDATE = int(UPDATE_DT / PRED_DT)
GYRO_RAD_TO_DEG = 180.0 / np.pi

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
    acc = acc / np.linalg.norm(acc)
    # Calculate pitch and roll
    pitch = np.arcsin(-acc[0])
    roll = np.arctan2(acc[1], acc[2])

    # Normalize magnetometer measurement
    mag = mag / np.linalg.norm(mag)
    # Compensate magnetometer readings with pitch and roll
    mx = mag[0]*np.cos(pitch) + mag[2]*np.sin(pitch)
    my = mag[0]*np.sin(roll)*np.sin(pitch) + mag[1]*np.cos(roll) - mag[2]*np.sin(roll)*np.cos(pitch)
    # Compute yaw
    yaw = np.arctan2(my, mx)

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

def rotation_matrix_to_quaternion(R):
    """Convert a 3x3 rotation matrix to a (w, x, y, z) quaternion."""
    m00, m01, m02 = R[0, :]
    m10, m11, m12 = R[1, :]
    m20, m21, m22 = R[2, :]

    tr = m00 + m11 + m22

    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2  # S=4*qw
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2  # S=4*qx
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2  # S=4*qy
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2  # S=4*qz
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S

    return np.array([qw, qx, qy, qz])

def compute_initial_quaternion(accel, mag):
    """
    Compute an initial quaternion from accelerometer and magnetometer readings,
    following PX4's convention: body frame is X-forward, Y-right, Z-down.
    accel, mag are in body frame.
    """

    # Normalize accelerometer -> 'down' in body frame
    down = accel / np.linalg.norm(accel)
    if down[2] < 0:  # since gravity-reaction force pointing up in body frame
        down = -down

    # Normalize magnetometer -> magnetic north (in body frame)
    north = mag / np.linalg.norm(mag)

    # Compute east in body frame
    east = np.cross(north, down)
    east /= np.linalg.norm(east)

    # Recompute north to ensure orthogonality
    north = np.cross(down, east)
    north /= np.linalg.norm(north)

    # --- Build rotation matrix (body->world) with PX4 axis conventions ---
    # PX4: x-forward, y-right, z-down
    forward = north          # body X points toward north initially
    right   = east           # body Y points east
    down    = down           # body Z points downward

    R = np.column_stack((forward, right, down))

    # Convert to quaternion (Hamilton, w,x,y,z)
    q_init = rotation_matrix_to_quaternion(R)
    return q_init

def integrate_gyro_euler(q_prev, gyro, dt):
    """
    Integrate gyro rates using Euler angles:
    - Convert q_prev (quaternion) to Euler angles (roll, pitch, yaw)
    - Add gyro * dt to each angle
    - Convert back to quaternion and return
    """
    # q_prev: [qw, qx, qy, qz]
    # gyro: [gx, gy, gz] in rad/s

    q_w, q_x, q_y, q_z = q_prev
    w_x, w_y, w_z = gyro

    q_w_next = dt*(q_x*(- 0.5*w_x) + q_y*(- 0.5*w_y) + q_z*(- 0.5*w_z)) + q_w
    q_x_next = dt*(q_w*(0.5*w_x) + q_y*(0.5*w_z) + q_z*(- 0.5*w_y)) + q_x
    q_y_next = dt*(q_w*(0.5*w_y) + q_x*(- 0.5*w_z) + q_z*(0.5*w_x)) + q_y 
    q_z_next = dt*(q_w*(0.5*w_z) + q_x*(0.5*w_y) + q_y*(- 0.5*w_x)) + q_z

    quat_new = np.array([q_w_next, q_x_next, q_y_next, q_z_next])
    return quat_new

def get_latest_index(timestamps, curr_time):
    """
    Returns the index of the latest timestamp at or before curr_time.
    """
    idx = np.searchsorted(timestamps, curr_time, side='right') - 1
    idx = max(idx, 0)
    return idx

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

    # ---- PX4 Estimator Attitude ----
    att_topic = next(x for x in ulog.data_list if x.name == 'vehicle_attitude')
    t_att = att_topic.data['timestamp'] * 1e-6
    qw_px4 = att_topic.data['q[0]']
    qx_px4 = att_topic.data['q[1]']
    qy_px4 = att_topic.data['q[2]']
    qz_px4 = att_topic.data['q[3]']

    # ---- EKF & Filters Init ----
    trim_time = 0.5 # seconds
    init_time = 5 # seconds

    # Calculate start_idx for each sensor based on its dt
    dt_mag = timestamps_mag[1] - timestamps_mag[0]
    dt_combined = timestamps_combined[1] - timestamps_combined[0]
    dt_baro = timestamps_baro[1] - timestamps_baro[0]

    trim_idx_mag = int(trim_time / dt_mag)
    trim_idx_combined = int(trim_time / dt_combined)
    trim_idx_baro = int(trim_time / dt_baro)

    start_idx_mag = int(init_time / dt_mag)
    start_idx_combined = int(init_time / dt_combined)
    start_idx_baro = int(init_time / dt_baro)

    start_time = max(
        timestamps_mag[start_idx_mag],
        timestamps_combined[start_idx_combined],
        timestamps_baro[start_idx_baro]
    )
    end_time = min(
        timestamps_mag[-1],
        timestamps_combined[-1],
        timestamps_baro[-1]
    )
    time_steps = np.arange(start_time, end_time, UPDATE_DT)

    mag_idx = start_idx_mag
    combined_idx = start_idx_combined
    baro_idx = start_idx_baro

    print("EKF boot up")

    # Initialize Filters
    velocity_z_low_pass_filter = LowPassFilter_1D(alpha=VEL_Z_LPF_ALPHA)
    accelerometer_low_pass_filter = LowPassFilter_3D(
        alpha=None,
        alpha_arr=[ACCEL_LPF_ALPHA_X, ACCEL_LPF_ALPHA_Y, ACCEL_LPF_ALPHA_Z],
    )
    magnetometer_low_pass_filter = LowPassFilter_3D(alpha=MAG_LPF_ALPHA)
    gyro_low_pass_filter = LowPassFilter_3D(alpha=GYRO_LPF_ALPHA)
    baro_low_pass_filter = LowPassFilter_1D(alpha=BARO_LPF_ALPHA)

    # Apply filters to initial samples and average for EKF init
    accel_samples = []
    gyro_samples = []
    mag_samples = []
    baro_samples = []

    for i in range(trim_idx_combined, start_idx_combined):
        accel_samples.append(accelerometer_low_pass_filter.update([
            sensor_combined_msgs.data['accelerometer_m_s2[0]'][i],
            sensor_combined_msgs.data['accelerometer_m_s2[1]'][i],
            sensor_combined_msgs.data['accelerometer_m_s2[2]'][i]
        ]))
        gyro_samples.append(gyro_low_pass_filter.update([
            sensor_combined_msgs.data['gyro_rad[0]'][i],
            sensor_combined_msgs.data['gyro_rad[1]'][i],
            sensor_combined_msgs.data['gyro_rad[2]'][i]
        ]))

    for i in range(trim_idx_mag, start_idx_mag):
        mag_samples.append(magnetometer_low_pass_filter.update([
            mag_msgs.data['x'][i],
            mag_msgs.data['y'][i],
            mag_msgs.data['z'][i]
        ]))

    for i in range(trim_idx_baro, start_idx_baro):
        baro_samples.append([
            baro_low_pass_filter.update(baro_msgs.data['pressure'][i]),
            baro_msgs.data['temperature'][i]
        ])

    init_accel = np.mean(np.array(accel_samples), axis=0)
    init_gyro = np.mean(np.array(gyro_samples), axis=0)
    init_gyro *= GYRO_RAD_TO_DEG
    init_mag = np.mean(np.array(mag_samples), axis=0)
    init_baro = np.mean(np.array(baro_samples), axis=0)

    init_quat = compute_initial_quaternion(init_accel, init_mag)
    euler = R.from_quat([init_quat[1], init_quat[2], init_quat[3], init_quat[0]]).as_euler('xyz', degrees=True) # scipy uses x,y,z,w
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
        # Prediction steps
        for _ in range(PRED_STEPS_PER_UPDATE):
            curr_time += PRED_DT
            orientation_full.predict()
            position_full.update_orientation(orientation_full.current_state[:4])
            position_full.predict()
            position_full.current_state[5] = velocity_z_low_pass_filter.update(
                position_full.current_state[5]
            )
            # orientation_full.update_ekf_history(time=curr_time)
            # position_full.update_ekf_history(time=curr_time)

        # Find latest sensor readings at or before curr_time
        mag_idx = get_latest_index(timestamps_mag, curr_time)
        combined_idx = get_latest_index(timestamps_combined, curr_time)
        baro_idx = get_latest_index(timestamps_baro, curr_time)
        # Filter Sensor Data
        accelerometer_filtered = accelerometer_low_pass_filter.update(
            [sensor_combined_msgs.data['accelerometer_m_s2[0]'][combined_idx],
             sensor_combined_msgs.data['accelerometer_m_s2[1]'][combined_idx],
             sensor_combined_msgs.data['accelerometer_m_s2[2]'][combined_idx]]
        )
        magnetometer_filtered = magnetometer_low_pass_filter.update(
            [mag_msgs.data['x'][mag_idx],
             mag_msgs.data['y'][mag_idx],
             mag_msgs.data['z'][mag_idx]]
        )
        gyro_filtered = gyro_low_pass_filter.update(
            [sensor_combined_msgs.data['gyro_rad[0]'][combined_idx],
             sensor_combined_msgs.data['gyro_rad[1]'][combined_idx],
             sensor_combined_msgs.data['gyro_rad[2]'][combined_idx]]
        )
        baro_pressure_filtered = baro_low_pass_filter.update(
            baro_msgs.data['pressure'][baro_idx]
        )

        # Update Orientation EKF
        orientation_full.set_observation(
            np.array([
                accelerometer_filtered[0],
                accelerometer_filtered[1],
                accelerometer_filtered[2],
                magnetometer_filtered[0],
                magnetometer_filtered[1],
                magnetometer_filtered[2],
            ]),
            np.array(gyro_filtered)
        )
        orientation_full.update(timestep=i)
        orientation_full.update_ekf_history(time=curr_time)

        # Update Position EKF
        position_full.set_observation(
            np.array([baro_pressure_filtered]),
            np.array(accelerometer_filtered),
        )
        position_full.update_orientation(orientation_full.current_state[:4])
        position_full.update(timestep=i)
        position_full.current_state[5] = velocity_z_low_pass_filter.update(
            position_full.current_state[5]
        )
        position_full.update_ekf_history(time=curr_time)


    # Reset Filters between scenarios
    velocity_z_low_pass_filter.reset()
    accelerometer_low_pass_filter.reset()
    magnetometer_low_pass_filter.reset()
    gyro_low_pass_filter.reset()
    baro_low_pass_filter.reset()


    # ---- Scenario 2: Predict  ----
    start_time_0 = max(
        timestamps_mag[0],
        timestamps_combined[0],
        timestamps_baro[0]
    )
    for curr_time in np.arange(start_time_0, end_time, PRED_DT):
        # Find latest sensor readings at or before curr_time
        mag_idx = get_latest_index(timestamps_mag, curr_time)
        combined_idx = get_latest_index(timestamps_combined, curr_time)
        baro_idx = get_latest_index(timestamps_baro, curr_time)
        # Filter Sensor Data
        accelerometer_filtered = accelerometer_low_pass_filter.update(
            [sensor_combined_msgs.data['accelerometer_m_s2[0]'][combined_idx],
             sensor_combined_msgs.data['accelerometer_m_s2[1]'][combined_idx],
             sensor_combined_msgs.data['accelerometer_m_s2[2]'][combined_idx]]
        )
        gyro_filtered = gyro_low_pass_filter.update(
            [sensor_combined_msgs.data['gyro_rad[0]'][combined_idx],
             sensor_combined_msgs.data['gyro_rad[1]'][combined_idx],
             sensor_combined_msgs.data['gyro_rad[2]'][combined_idx]]
        )

        orientation_pred.current_input = np.array(gyro_filtered)
        position_pred.current_input = np.array(accelerometer_filtered)
        position_pred.update_orientation(orientation_pred.current_state[:4])

        # Direct quaternion integration instead of EKF predict
        orientation_pred.current_state[:4] = integrate_gyro_euler(
            orientation_pred.current_state[:4], gyro_filtered, PRED_DT
        )

        position_pred.predict()
        position_full.current_state[5] = velocity_z_low_pass_filter.update(
            position_full.current_state[5]
        )
        orientation_pred.update_ekf_history(time=curr_time)
        position_pred.update_ekf_history(time=curr_time)
        prev_time = curr_time


    # Reset Filters between scenarios
    velocity_z_low_pass_filter.reset()
    accelerometer_low_pass_filter.reset()
    magnetometer_low_pass_filter.reset()
    gyro_low_pass_filter.reset()
    baro_low_pass_filter.reset()


    # ---- Scenario 3: Update Only ----
    for curr_time in time_steps:
        # Find latest sensor readings at or before curr_time
        mag_idx = get_latest_index(timestamps_mag, curr_time)
        combined_idx = get_latest_index(timestamps_combined, curr_time)
        baro_idx = get_latest_index(timestamps_baro, curr_time)
        # Filter Sensor Data
        accelerometer_filtered = accelerometer_low_pass_filter.update(
            [sensor_combined_msgs.data['accelerometer_m_s2[0]'][combined_idx],
             sensor_combined_msgs.data['accelerometer_m_s2[1]'][combined_idx],
             sensor_combined_msgs.data['accelerometer_m_s2[2]'][combined_idx]]
        )
        magnetometer_filtered = magnetometer_low_pass_filter.update(
            [mag_msgs.data['x'][mag_idx],
             mag_msgs.data['y'][mag_idx],
             mag_msgs.data['z'][mag_idx]]
        )
        gyro_filtered = gyro_low_pass_filter.update(
            [sensor_combined_msgs.data['gyro_rad[0]'][combined_idx],
             sensor_combined_msgs.data['gyro_rad[1]'][combined_idx],
             sensor_combined_msgs.data['gyro_rad[2]'][combined_idx]]
        )
        baro_pressure_filtered = baro_low_pass_filter.update(
            baro_msgs.data['pressure'][baro_idx]
        )

        # Orientation Update - manual
        quaternion = np.array([acc_mag_to_quaternion(
            np.array(accelerometer_filtered),
            np.array(magnetometer_filtered)
        )])
        orientation_upd.current_state[:4] = quaternion
        orientation_upd.update_ekf_history(time=curr_time)

        # Position EKF Update — manual
        pz = (
            np.log(baro_pressure_filtered / position_upd.PRESSURE_SEA_LEVEL_Pa) *
            position_upd.ROOM_TEMPERATURE_K *
            position_upd.UNIVERSAL_GAS_CONSTANT_JmolK / (
                position_upd.AIR_MOLAR_MASS_kgmol *
                position_upd.GRAVITATION_ACCELERATION_ms2
            )
        ) # BAROMETRIC FORMULA
        position_upd.current_state[2] = pz
        position_upd.update_ekf_history(time=curr_time)

    # Reset Filters between scenarios
    velocity_z_low_pass_filter.reset()
    accelerometer_low_pass_filter.reset()
    magnetometer_low_pass_filter.reset()
    gyro_low_pass_filter.reset()
    baro_low_pass_filter.reset()


    # ---- Scenario 4: Full EKF + NN ----
    if with_nn:
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

        for curr_time in time_steps:
            # Prediction steps
            for _ in range(PRED_STEPS_PER_UPDATE):
                curr_time += PRED_DT
                orientation_full_nn.predict()
                position_full_nn.update_orientation(orientation_full_nn.current_state[:4])
                position_full_nn.predict()
                position_full_nn.current_state[5] = velocity_z_low_pass_filter.update(
                    position_full_nn.current_state[5]
                )
                orientation_full_nn.update_ekf_history(time=curr_time)
                position_full_nn.update_ekf_history(time=curr_time)

            # Find latest sensor readings at or before curr_time
            mag_idx = get_latest_index(timestamps_mag, curr_time)
            combined_idx = get_latest_index(
                timestamps_combined, curr_time
                )
            baro_idx = get_latest_index(timestamps_baro, curr_time)
            # Filter Sensor Data
            accelerometer_filtered = accelerometer_low_pass_filter.update(
                [sensor_combined_msgs.data['accelerometer_m_s2[0]'][combined_idx],
                 sensor_combined_msgs.data['accelerometer_m_s2[1]'][combined_idx],
                 sensor_combined_msgs.data['accelerometer_m_s2[2]'][combined_idx]]
            )
            magnetometer_filtered = magnetometer_low_pass_filter.update(
                [mag_msgs.data['x'][mag_idx],
                 mag_msgs.data['y'][mag_idx],
                 mag_msgs.data['z'][mag_idx]]
            )
            gyro_filtered = gyro_low_pass_filter.update(
                [sensor_combined_msgs.data['gyro_rad[0]'][combined_idx],
                 sensor_combined_msgs.data['gyro_rad[1]'][combined_idx],
                 sensor_combined_msgs.data['gyro_rad[2]'][combined_idx]]
            )
            gyro_filtered = [g * GYRO_RAD_TO_DEG for g in gyro_filtered]
            baro_pressure_filtered = baro_low_pass_filter.update(
                baro_msgs.data['pressure'][baro_idx]
            )

            # Neural Net Correction
            ekf_state = np.concat((
                position_full_nn.current_state, # pos, vel
                orientation_full_nn.current_state[:4] # quaternion
            ))
            nn_input_np = np.concat((
                ekf_state,
                accelerometer_filtered,
                gyro_filtered,
                [baro_pressure_filtered],
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
                np.array(gyro_filtered)
            )
            orientation_full_nn.update(timestep=i)
            orientation_full_nn.update_ekf_history(time=curr_time)

            # Update Position EKF only when new barometer data is available
            position_full_nn.set_observation(
                np.array([baro_pressure_filtered]),
                np.array(accelerometer_filtered),
            )
            position_full_nn.update_orientation(orientation_full_nn.current_state[:4])
            position_full_nn.update(timestep=i)
            position_full_nn.current_state[5] = velocity_z_low_pass_filter.update(
                position_full_nn.current_state[5]
            )
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
