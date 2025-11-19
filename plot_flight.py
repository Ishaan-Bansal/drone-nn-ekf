from pyulog import ULog
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from signal_filters import LowPassFilter
from parameters import (
    ACCEL_LPF_ALPHA_X, ACCEL_LPF_ALPHA_Y, ACCEL_LPF_ALPHA_Z,
    MAG_LPF_ALPHA, GYRO_LPF_ALPHA, BARO_LPF_ALPHA,
    TRAINING_FILES, TEST_FILES,
)

filename = TRAINING_FILES[1] 
ulog_file = f'flight_data/{filename}.ulg'
log = ULog(ulog_file)

def plot_sensor(topic_name, axes_labels, ylabel, title):
    fig, ax = plt.subplots(len(axes_labels), 1, figsize=(10, 9), sharex=True)
    topic = next((x for x in log.data_list if x.name == topic_name), None)
    if topic is not None:
        t = topic.data['timestamp'] * 1e-6  # seconds
        print(f"Average timestep ({ylabel})= ", np.mean(np.diff(t)))
        if len(axes_labels)==1:
            ax.plot(t, topic.data[axes_labels[0]], "x", label=axes_labels[0])
            ax.set_ylabel(f"{ylabel} ({axes_labels[0]})")
            ax.legend()
            ax.grid()
            ax.set_xlabel('Time [s]')
        else :
            for idx, label in enumerate(axes_labels):
                ax[idx].plot(t, topic.data[label], "x", label=label)
                ax[idx].set_ylabel(f"{ylabel} ({label})")
                ax[idx].legend()
                ax[idx].grid()
            ax[-1].set_xlabel('Time [s]')
        fig.suptitle(title)
        plt.tight_layout()

# --- Sensor Plots ---
sensor_combined_topic = next((t for t in log.data_list if t.name == 'sensor_combined'), None)
sensor_mag_topic = next((t for t in log.data_list if t.name == 'sensor_mag'), None)
sensor_baro_topic = next((t for t in log.data_list if t.name == 'sensor_baro'), None)

# Use sensor_combined for accel, gyro; sensor_mag for mag; sensor_baro for baro
plot_sensor('sensor_combined', ['accelerometer_m_s2[0]', 'accelerometer_m_s2[1]', 'accelerometer_m_s2[2]'], r'Accel [m/$s^2$]', 'Accelerometer')
plot_sensor('sensor_combined', ['gyro_rad[0]', 'gyro_rad[1]', 'gyro_rad[2]'], 'Gyro [rad/s]', 'Gyroscope')
plot_sensor('sensor_mag', ['x', 'y', 'z'], 'Mag [Gauss]', 'Magnetometer')
plot_sensor('sensor_baro', ['pressure'], 'Pressure (Pa)', 'Barometer')

# Filtered sensor data
def plot_raw_and_filtered_3d_sensor_data(sensor_topic, time_array, filter_instance, axis_labels, sensor_name, alpha_label="alpha"):
    """
    Apply a 3D online low-pass filter to sensor data and plot raw vs filtered data.

    Parameters:
    - sensor_topic: ULog topic object with 3D sensor data ('x', 'y', 'z')
    - time_array: numpy array of timestamps corresponding to sensor data
    - filter_instance: an instance of a 3D low-pass filter class with an update() method
    - axis_labels: list of three strings ['x', 'y', 'z'] representing the sensor data fields
    - sensor_name: string, friendly name for plot titles and labels (e.g. 'Accelerometer')
    - alpha_label: string, label of filter parameter alpha for the legend (optional)

    Returns:
    - filtered_data numpy array with shape (N,3) of filtered sensor data
    """

    # Extract raw sensor data for x, y, z
    x_data = sensor_topic.data[axis_labels[0]]
    y_data = sensor_topic.data[axis_labels[1]]
    z_data = sensor_topic.data[axis_labels[2]]

    filtered_data = []
    for x_val, y_val, z_val in zip(x_data, y_data, z_data):
        filtered_point = filter_instance.update([x_val, y_val, z_val])
        filtered_data.append(filtered_point)
    filtered_data = np.array(filtered_data)

    # Plot raw and filtered data per axis
    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for i, axis in enumerate(axis_labels):
        ax[i].plot(time_array, sensor_topic.data[axis], label=f'Raw {axis}')
        ax[i].plot(time_array, filtered_data[:, i], label=f'Filtered {axis} ({alpha_label}={filter_instance.alpha})')
        ax[i].set_ylabel(f'{sensor_name} Magnitude')
        ax[i].legend()
        ax[i].grid()
    ax[-1].set_xlabel('Time [s]')
    fig.suptitle(f'{sensor_name} Raw vs Filtered')
    plt.tight_layout()
    return filtered_data

accelerometer_low_pass_filter_orientation = LowPassFilter(
    alpha=np.array([ACCEL_LPF_ALPHA_X, ACCEL_LPF_ALPHA_Y, ACCEL_LPF_ALPHA_Z]),
)
magnetometer_low_pass_filter = LowPassFilter(alpha=np.ones(3)*MAG_LPF_ALPHA)
gyro_low_pass_filter = LowPassFilter(alpha=np.ones(3)*GYRO_LPF_ALPHA)
baro_low_pass_filter = LowPassFilter(alpha=np.ones(1)*BARO_LPF_ALPHA)

# For accelerometer orientation filtered data
accel_filtered = plot_raw_and_filtered_3d_sensor_data(
    sensor_combined_topic,
    sensor_combined_topic.data['timestamp'] * 1e-6,
    accelerometer_low_pass_filter_orientation,
    ['accelerometer_m_s2[0]', 'accelerometer_m_s2[1]', 'accelerometer_m_s2[2]'],
    'Accelerometer',
    f"alpha={[ACCEL_LPF_ALPHA_X, ACCEL_LPF_ALPHA_Y, ACCEL_LPF_ALPHA_Z]}"
)

mag_filtered = plot_raw_and_filtered_3d_sensor_data(
    sensor_mag_topic,
    sensor_mag_topic.data['timestamp'] * 1e-6,
    magnetometer_low_pass_filter,
    ['x', 'y', 'z'],
    'Magnetometer',
    f"alpha={MAG_LPF_ALPHA}"
)

gyro_filtered = plot_raw_and_filtered_3d_sensor_data(
    sensor_combined_topic,
    sensor_combined_topic.data['timestamp'] * 1e-6,
    gyro_low_pass_filter,
    ['gyro_rad[0]', 'gyro_rad[1]', 'gyro_rad[2]'],
    'Gyroscope',
    f"alpha={GYRO_LPF_ALPHA}"
)

# Filter barometer pressure (1D)
pressure = sensor_baro_topic.data['pressure'] # Pascals
t_baro = sensor_baro_topic.data['timestamp'] * 1e-6
print("Average timestep (BARO)= ", np.mean(np.diff(t_baro)))
pressure_filtered = []
for p in pressure:
    pressure_filtered.append(baro_low_pass_filter.update(np.array([p])))

fig_baro_pressure_filtered = plt.figure()
plt.plot(t_baro, pressure, label='Raw Pressure')
plt.plot(t_baro, pressure_filtered, label=f'Filtered Pressure (alpha={BARO_LPF_ALPHA})')
plt.xlabel('Time [s]')
plt.ylabel('Pressure (Pa)')
plt.title('Barometer Pressure Raw vs Filtered')
plt.legend()
plt.grid()
plt.tight_layout()

# --- State Only (no setpoints for pos/vel) ---
pos_topic = next(x for x in log.data_list if x.name == 'vehicle_local_position')
t = pos_topic.data['timestamp'] * 1e-6
print("Average timestep (POS EKF)= ", np.mean(np.diff(t)))
x = pos_topic.data['x']
y = pos_topic.data['y']
z = pos_topic.data['z']
vx = pos_topic.data['vx']
vy = pos_topic.data['vy']
vz = pos_topic.data['vz']

# X position/velocity
fig_xpos = plt.figure()
ax1 = fig_xpos.add_subplot(211)
ax1.plot(t, x, label='x')
ax1.set_ylabel('X position [m]')
ax1.grid(), ax1.legend()
ax1.set_title('X Position')
ax2 = fig_xpos.add_subplot(212)
ax2.plot(t, vx, label='vx')
ax2.set_ylabel('X velocity [m/s]')
ax2.set_xlabel('Time [s]')
ax2.grid(), ax2.legend()
ax2.set_title('X Velocity')
plt.tight_layout()

# Y position/velocity
fig_ypos = plt.figure()
ax1 = fig_ypos.add_subplot(211)
ax1.plot(t, y, label='y')
ax1.set_ylabel('Y position [m]')
ax1.grid(), ax1.legend()
ax1.set_title('Y Position')
ax2 = fig_ypos.add_subplot(212)
ax2.plot(t, vy, label='vy')
ax2.set_ylabel('Y velocity [m/s]')
ax2.set_xlabel('Time [s]')
ax2.grid(), ax2.legend()
ax2.set_title('Y Velocity')
plt.tight_layout()

# Z position/velocity
fig_zpos = plt.figure()
ax1 = fig_zpos.add_subplot(211)
ax1.plot(t, z, label='z')
ax1.set_ylabel('Z position [m]')
ax1.grid(), ax1.legend()
ax1.set_title('Z Position')
ax2 = fig_zpos.add_subplot(212)
ax2.plot(t, vz, label='vz')
ax2.set_ylabel('Z velocity [m/s]')
ax2.set_xlabel('Time [s]')
ax2.grid(), ax2.legend()
ax2.set_title('Z Velocity')
plt.tight_layout()

# --- Attitude (Quaternion + Euler) and Angular Setpoints ---

att_topic = next(x for x in log.data_list if x.name == 'vehicle_attitude')
att_sp_topic = next((x for x in log.data_list if x.name == 'vehicle_attitude_setpoint'), None)

t_att = att_topic.data['timestamp'] * 1e-6
print("Average timestep (QUAT EKF)= ", np.mean(np.diff(t_att)))
q0 = att_topic.data['q[0]']
q1 = att_topic.data['q[1]']
q2 = att_topic.data['q[2]']
q3 = att_topic.data['q[3]']

def quaternion_to_euler(w, x, y, z):
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw

roll, pitch, yaw = quaternion_to_euler(q0, q1, q2, q3)

if att_sp_topic is not None:
    t_sp = att_sp_topic.data['timestamp'] * 1e-6
    q0_sp = att_sp_topic.data['q_d[0]']
    q1_sp = att_sp_topic.data['q_d[1]']
    q2_sp = att_sp_topic.data['q_d[2]']
    q3_sp = att_sp_topic.data['q_d[3]']
    roll_sp, pitch_sp, yaw_sp = quaternion_to_euler(q0_sp, q1_sp, q2_sp, q3_sp)
    # Interpolate setpoints onto actual timebase
    roll_sp_interp = np.interp(t_att, t_sp, roll_sp)
    pitch_sp_interp = np.interp(t_att, t_sp, pitch_sp)
    yaw_sp_interp = np.interp(t_att, t_sp, yaw_sp)
else:
    roll_sp_interp = pitch_sp_interp = yaw_sp_interp = np.full_like(roll, np.nan)

fig_euler = plt.figure()
# Roll
ax1 = fig_euler.add_subplot(311)
ax1.plot(t_att, np.degrees(roll), label='roll')
ax1.plot(t_att, np.degrees(roll_sp_interp), label='roll_setpoint', linestyle='--')
ax1.set_ylabel('Roll [deg]')
ax1.grid(), ax1.legend()
ax1.set_title('Roll and Setpoint')
# Pitch
ax2 = fig_euler.add_subplot(312)
ax2.plot(t_att, np.degrees(pitch), label='pitch')
ax2.plot(t_att, np.degrees(pitch_sp_interp), label='pitch_setpoint', linestyle='--')
ax2.set_ylabel('Pitch [deg]')
ax2.grid(), ax2.legend()
ax2.set_title('Pitch and Setpoint')
# Yaw
ax3 = fig_euler.add_subplot(313)
ax3.plot(t_att, np.degrees(yaw), label='yaw')
ax3.plot(t_att, np.degrees(yaw_sp_interp), label='yaw_setpoint', linestyle='--')
ax3.set_ylabel('Yaw [deg]')
ax3.set_xlabel('Time [s]')
ax3.grid(), ax3.legend()
ax3.set_title('Yaw and Setpoint')
plt.tight_layout()

# --- Integrate filtered gyro data to estimate Euler angles ---
gyro_t = sensor_combined_topic.data['timestamp'] * 1e-6
dt_gyro = np.diff(gyro_t, prepend=gyro_t[0])
gyro_roll = np.zeros_like(gyro_t)
gyro_pitch = np.zeros_like(gyro_t)
gyro_yaw = np.zeros_like(gyro_t)

for i in range(1, len(gyro_t)):
    gyro_roll[i] = gyro_roll[i-1] + gyro_filtered[i-1, 0] * dt_gyro[i]
    gyro_pitch[i] = gyro_pitch[i-1] + gyro_filtered[i-1, 1] * dt_gyro[i]
    gyro_yaw[i] = gyro_yaw[i-1] + gyro_filtered[i-1, 2] * dt_gyro[i]

# Convert to degrees for plotting
gyro_roll_deg = np.degrees(gyro_roll)
gyro_pitch_deg = np.degrees(gyro_pitch)
gyro_yaw_deg = np.degrees(gyro_yaw)

# --- Add to Euler angle plot ---
# Roll
ax1.plot(gyro_t, gyro_roll_deg, label='Gyro Integrated Roll', linestyle=':')
# Pitch
ax2.plot(gyro_t, gyro_pitch_deg, label='Gyro Integrated Pitch', linestyle=':')
# Yaw
ax3.plot(gyro_t, gyro_yaw_deg, label='Gyro Integrated Yaw', linestyle=':')

ax1.legend()
ax2.legend()
ax3.legend()

# --- 3D Trajectory with Time Gradient ---
t_norm = (t - np.min(t)) / (np.max(t) - np.min(t))
fig_traj = plt.figure()
ax = fig_traj.add_subplot(111, projection='3d')
sc = ax.scatter(x, y, z, c=t_norm, cmap='viridis', marker='o', s=5)
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.set_title('3D Trajectory with Time Gradient')
cbar = plt.colorbar(sc, ax=ax, pad=0.1)
cbar.set_label('Normalized Time (0=start, 1=end)')
plt.tight_layout()

plt.show()
