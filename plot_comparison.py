from pyulog import ULog
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from parameters import TRAINING_FILES, TEST_FILES

filename = TRAINING_FILES[0]
WITH_NN = False  # Set to True if comparing EKF + NN

# --- Load PX4 ULOG ---
ulog_file = f'flight_data\{filename}.ulg'
log = ULog(ulog_file)

# --- PX4 Estimator Attitude ---
att_topic = next(x for x in log.data_list if x.name == 'vehicle_attitude')
t_att = att_topic.data['timestamp'] * 1e-6
qw_px4 = att_topic.data['q[0]']
qx_px4 = att_topic.data['q[1]']
qy_px4 = att_topic.data['q[2]']
qz_px4 = att_topic.data['q[3]']

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

# PX4 Euler angles
roll_px4, pitch_px4, yaw_px4 = quaternion_to_euler(qw_px4, qx_px4, qy_px4, qz_px4)

# --- Load EKF outputs for each scenario ---
orientation_full = pd.read_csv(f'./EKF_data/{filename}_orientation_ekf_full_states.csv')
orientation_pred = pd.read_csv(f'./EKF_data/{filename}_orientation_ekf_pred_only_states.csv')
orientation_upd = pd.read_csv(f'./EKF_data/{filename}_orientation_ekf_update_only_states.csv')
position_full = pd.read_csv(f'./EKF_data/{filename}_position_ekf_full_states.csv')
position_pred = pd.read_csv(f'./EKF_data/{filename}_position_ekf_pred_only_states.csv')
position_upd = pd.read_csv(f'./EKF_data/{filename}_position_ekf_update_only_states.csv')
if WITH_NN:
    orientation_full_nn = pd.read_csv(f'./EKF_data/{filename}_orientation_ekf_full_nn_states.csv')
    position_full_nn = pd.read_csv(f'./EKF_data/{filename}_position_ekf_full_nn_states.csv')


def extract_quat_and_time(df):
    return (
        df['quaternion (w)'].values,
        df['quaternion (x)'].values,
        df['quaternion (y)'].values,
        df['quaternion (z)'].values,
        df['time'].values
    )

qw_full, qx_full, qy_full, qz_full, time_full = extract_quat_and_time(orientation_full)
qw_pred, qx_pred, qy_pred, qz_pred, time_pred = extract_quat_and_time(orientation_pred)
qw_upd, qx_upd, qy_upd, qz_upd, time_upd = extract_quat_and_time(orientation_upd)
if WITH_NN:
    qw_nn, qx_nn, qy_nn, qz_nn, time_nn = extract_quat_and_time(orientation_full_nn)
    roll_nn, pitch_nn, yaw_nn = quaternion_to_euler(qw_nn, qx_nn, qy_nn, qz_nn)

# Convert EKF quaternions to Euler angles (no offset or wrapping needed)
roll_full, pitch_full, yaw_full = quaternion_to_euler(qw_full, qx_full, qy_full, qz_full)
roll_pred, pitch_pred, yaw_pred = quaternion_to_euler(qw_pred, qx_pred, qy_pred, qz_pred)
roll_upd, pitch_upd, yaw_upd = quaternion_to_euler(qw_upd, qx_upd, qy_upd, qz_upd)

colors = {
    'PX4': 'blue',
    'EKF Full': 'yellow',
    'Prediction Only': 'magenta',
    'Measurement Only': 'black',
    'EKF + NN': 'lime'
}

# --- Plot Quaternion Components Comparison ---
fig_quat_comp, axs = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
axs[0].plot(t_att, qw_px4, label='PX4 qw', color=colors['PX4'])
axs[0].plot(time_full, qw_full, label='EKF Full', linestyle='-', color=colors['EKF Full'])
axs[0].plot(time_pred, qw_pred, label='Prediction Only', linestyle='--', color=colors['Prediction Only'])
axs[0].plot(time_upd, qw_upd, label='Measurement Only', linestyle=':', color=colors['Measurement Only'])
if WITH_NN: axs[0].plot(time_nn, qw_nn, label='EKF + NN', linestyle='-.', color=colors['EKF + NN'])
axs[0].set_ylabel('qw')
axs[0].legend()
axs[0].grid()

axs[1].plot(t_att, qx_px4, label='PX4 qx', color=colors['PX4'])
axs[1].plot(time_full, qx_full, label='EKF Full', linestyle='-', color=colors['EKF Full'])
axs[1].plot(time_pred, qx_pred, label='Prediction Only', linestyle='--', color=colors['Prediction Only'])
axs[1].plot(time_upd, qx_upd, label='Measurement Only', linestyle=':', color=colors['Measurement Only'])
if WITH_NN: axs[1].plot(time_nn, qx_nn, label='EKF + NN', linestyle='-.', color=colors['EKF + NN'])
axs[1].set_ylabel('qx')
axs[1].legend()
axs[1].grid()

axs[2].plot(t_att, qy_px4, label='PX4 qy', color=colors['PX4'])
axs[2].plot(time_full, qy_full, label='EKF Full', linestyle='-', color=colors['EKF Full'])
axs[2].plot(time_pred, qy_pred, label='Prediction Only', linestyle='--', color=colors['Prediction Only'])
axs[2].plot(time_upd, qy_upd, label='Measurement Only', linestyle=':', color=colors['Measurement Only'])
if WITH_NN: axs[2].plot(time_nn, qy_nn, label='EKF + NN', linestyle='-.', color=colors['EKF + NN'])
axs[2].set_ylabel('qy')
axs[2].legend()
axs[2].grid()

axs[3].plot(t_att, qz_px4, label='PX4 qz', color=colors['PX4'])
axs[3].plot(time_full, qz_full, label='EKF Full', linestyle='-', color=colors['EKF Full'])
axs[3].plot(time_pred, qz_pred, label='Prediction Only', linestyle='--', color=colors['Prediction Only'])
axs[3].plot(time_upd, qz_upd, label='Measurement Only', linestyle=':', color=colors['Measurement Only'])
if WITH_NN: axs[3].plot(time_nn, qz_nn, label='EKF + NN', linestyle='-.', color=colors['EKF + NN'])
axs[3].set_ylabel('qz')
axs[3].set_xlabel('Time [s]')
axs[3].legend()
axs[3].grid()
fig_quat_comp.suptitle('Quaternion Comparison: PX4 Estimator vs EKF Scenarios')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# --- Plot Euler Angle Comparison ---
fig_euler_comp = plt.figure(figsize=(10, 8))
ax1 = fig_euler_comp.add_subplot(311)
ax1.plot(t_att, np.degrees(roll_px4), label='PX4 Roll', color=colors['PX4'])
ax1.plot(time_full, np.degrees(roll_full), label='EKF Full', linestyle='-', color=colors['EKF Full'])
ax1.plot(time_pred, np.degrees(roll_pred), label='Prediction Only', linestyle='--', color=colors['Prediction Only'])
ax1.plot(time_upd, np.degrees(roll_upd), label='Measurement Only', linestyle=':', color=colors['Measurement Only'])
if WITH_NN:
    ax1.plot(time_nn, np.degrees(roll_nn), label='EKF + NN', linestyle='-.', color=colors['EKF + NN'])
ax1.set_ylabel('Roll [deg]')
ax1.grid()
ax1.legend()
ax1.set_title('Roll Comparison')

ax2 = fig_euler_comp.add_subplot(312)
ax2.plot(t_att, np.degrees(pitch_px4), label='PX4 Pitch', color=colors['PX4'])
ax2.plot(time_full, np.degrees(pitch_full), label='EKF Full', linestyle='-', color=colors['EKF Full'])
ax2.plot(time_pred, np.degrees(pitch_pred), label='Prediction Only', linestyle='--', color=colors['Prediction Only'])
ax2.plot(time_upd, np.degrees(pitch_upd), label='Measurement Only', linestyle=':', color=colors['Measurement Only'])
if WITH_NN: 
    ax2.plot(time_nn, np.degrees(pitch_nn), label='EKF + NN', linestyle='-.', color=colors['EKF + NN'])
ax2.set_ylabel('Pitch [deg]')
ax2.grid()
ax2.legend()
ax2.set_title('Pitch Comparison')

ax3 = fig_euler_comp.add_subplot(313)
ax3.plot(t_att, np.degrees(yaw_px4), label='PX4 Yaw (offset zeroed)', color=colors['PX4'])
ax3.plot(time_full, np.degrees(yaw_full), label='EKF Full', linestyle='-', color=colors['EKF Full'])
ax3.plot(time_pred, np.degrees(yaw_pred), label='Prediction Only', linestyle='--', color=colors['Prediction Only'])
ax3.plot(time_upd, np.degrees(yaw_upd), label='Measurement Only', linestyle=':', color=colors['Measurement Only'])
if WITH_NN: 
    ax3.plot(time_nn, np.degrees(yaw_nn), label='EKF + NN', linestyle='-.', color=colors['EKF + NN'])
ax3.set_ylabel('Yaw [deg]')
ax3.set_xlabel('Time [s]')
ax3.grid()
ax3.legend()
ax3.set_title('Yaw Comparison (initial PX4 offset removed)')
plt.tight_layout()

# --- Position and Velocity comparison ---
pos_topic = next(x for x in log.data_list if x.name == 'vehicle_local_position')
t = pos_topic.data['timestamp'] * 1e-6
x_px4 = pos_topic.data['x']
y_px4 = pos_topic.data['y']
z_px4 = pos_topic.data['z']
vx_px4 = pos_topic.data['vx']
vy_px4 = pos_topic.data['vy']
vz_px4 = pos_topic.data['vz']

# Remove offset
x_px4 -= x_px4[0]
y_px4 -= y_px4[0]
z_px4 -= z_px4[0]

def extract_pos_vel(df):
    x = df['position (x) [m]'].values if 'position (x) [m]' in df.columns else None
    y = df['position (y) [m]'].values if 'position (y) [m]' in df.columns else None
    z = df['position (z) [m]'].values if 'position (z) [m]' in df.columns else None
    vx = df['velocity (x) [m/s]'].values if 'velocity (x) [m/s]' in df.columns else None
    vy = df['velocity (y) [m/s]'].values if 'velocity (y) [m/s]' in df.columns else None
    vz = df['velocity (z) [m/s]'].values if 'velocity (z) [m/s]' in df.columns else None
    time = df['time'].values if 'time' in df.columns else None
    return x, y, z, vx, vy, vz, time

x_full, y_full, z_full, vx_full, vy_full, vz_full, time_full_pos = extract_pos_vel(position_full)
x_pred, y_pred, z_pred, vx_pred, vy_pred, vz_pred, time_pred_pos = extract_pos_vel(position_pred)
x_upd, y_upd, z_upd, vx_upd, vy_upd, vz_upd, time_upd_pos = extract_pos_vel(position_upd)
if WITH_NN:
    x_nn, y_nn, z_nn, vx_nn, vy_nn, vz_nn, time_nn_pos = extract_pos_vel(position_full_nn)

from matplotlib.gridspec import GridSpec

fig_pos = plt.figure(figsize=(12, 10))
gs = GridSpec(3, 2, figure=fig_pos)

ax_pos_x = fig_pos.add_subplot(gs[0, 0])
ax_pos_x.plot(t, x_px4, label='PX4 X', color=colors['PX4'])
ax_pos_x.plot(time_full_pos, x_full, label='EKF Full', linestyle='-', color=colors['EKF Full'])
ax_pos_x.plot(time_pred_pos, x_pred, label='Prediction Only', linestyle='--', color=colors['Prediction Only'])
ax_pos_x.plot(time_upd_pos, x_upd, label='Measurement Only', linestyle=':', color=colors['Measurement Only'])
if WITH_NN: ax_pos_x.plot(time_nn_pos, x_nn, label='EKF + NN', linestyle='-.', color=colors['EKF + NN'])
ax_pos_x.set_ylabel('X Position [m]')
ax_pos_x.grid()
ax_pos_x.legend()
ax_pos_x.set_title('X Position Comparison')

ax_pos_y = fig_pos.add_subplot(gs[1, 0])
ax_pos_y.plot(t, y_px4, label='PX4 Y', color=colors['PX4'])
ax_pos_y.plot(time_full_pos, y_full, label='EKF Full', linestyle='-', color=colors['EKF Full'])
ax_pos_y.plot(time_pred_pos, y_pred, label='Prediction Only', linestyle='--', color=colors['Prediction Only'])
ax_pos_y.plot(time_upd_pos, y_upd, label='Measurement Only', linestyle=':', color=colors['Measurement Only']) 
if WITH_NN: 
    ax_pos_y.plot(time_nn_pos, y_nn, label='EKF + NN', linestyle='-.', color=colors['EKF + NN']) # Blows up
ax_pos_y.set_ylabel('Y Position [m]')
ax_pos_y.grid()
ax_pos_y.legend()
ax_pos_y.set_title('Y Position Comparison')

ax_pos_z = fig_pos.add_subplot(gs[2, 0])
ax_pos_z.plot(t, z_px4, label='PX4 Z', color=colors['PX4'])
ax_pos_z.plot(time_full_pos, z_full, label='EKF Full', linestyle='-', color=colors['EKF Full'])
ax_pos_z.plot(time_pred_pos, z_pred, label='Prediction Only', linestyle='--', color=colors['Prediction Only'])
ax_pos_z.plot(time_upd_pos, z_upd, label='Measurement Only', linestyle=':', color=colors['Measurement Only'])
if WITH_NN: 
    ax_pos_z.plot(time_nn_pos, z_nn, label='EKF + NN', linestyle='-.', color=colors['EKF + NN']) # Blows up
ax_pos_z.set_ylabel('Z Position [m]')
ax_pos_z.set_xlabel('Time [s]')
ax_pos_z.grid()
ax_pos_z.legend()
ax_pos_z.set_title('Z Position Comparison')

ax_vel_x = fig_pos.add_subplot(gs[0, 1])
ax_vel_x.plot(t, vx_px4, label='PX4 Vx', color=colors['PX4'])
ax_vel_x.plot(time_full_pos, vx_full, label='EKF Full', linestyle='-', color=colors['EKF Full'])
ax_vel_x.plot(time_pred_pos, vx_pred, label='Prediction Only', linestyle='--', color=colors['Prediction Only'])
ax_vel_x.plot(time_upd_pos, vx_upd, label='Measurement Only', linestyle=':', color=colors['Measurement Only'])
if WITH_NN: 
    ax_vel_x.plot(time_nn_pos, vx_nn, label='EKF + NN', linestyle='-.', color=colors['EKF + NN']) # Blows up
ax_vel_x.set_ylabel('X Velocity [m/s]')
ax_vel_x.grid()
ax_vel_x.legend()
ax_vel_x.set_title('X Velocity Comparison')

ax_vel_y = fig_pos.add_subplot(gs[1, 1])
ax_vel_y.plot(t, vy_px4, label='PX4 Vy', color=colors['PX4'])
ax_vel_y.plot(time_full_pos, vy_full, label='EKF Full', linestyle='-', color=colors['EKF Full'])
ax_vel_y.plot(time_pred_pos, vy_pred, label='Prediction Only', linestyle='--', color=colors['Prediction Only'])
ax_vel_y.plot(time_upd_pos, vy_upd, label='Measurement Only', linestyle=':', color=colors['Measurement Only'])
if WITH_NN: 
    ax_vel_y.plot(time_nn_pos, vy_nn, label='EKF + NN', linestyle='-.', color=colors['EKF + NN']) # Blows up
ax_vel_y.set_ylabel('Y Velocity [m/s]')
ax_vel_y.grid()
ax_vel_y.legend()
ax_vel_y.set_title('Y Velocity Comparison')

ax_vel_z = fig_pos.add_subplot(gs[2, 1])
ax_vel_z.plot(t, vz_px4, label='PX4 Vz', color=colors['PX4'])
ax_vel_z.plot(time_full_pos, vz_full, label='EKF Full', linestyle='-', color=colors['EKF Full'])
ax_vel_z.plot(time_pred_pos, vz_pred, label='Prediction Only', linestyle='--', color=colors['Prediction Only'])
ax_vel_z.plot(time_upd_pos, vz_upd, label='Measurement Only', linestyle=':', color=colors['Measurement Only'])
if WITH_NN: 
    ax_vel_z.plot(time_nn_pos, vz_nn, label='EKF + NN', linestyle='-.', color=colors['EKF + NN']) # Blows up
ax_vel_z.set_ylabel('Z Velocity [m/s]')
ax_vel_z.set_xlabel('Time [s]')
ax_vel_z.grid()
ax_vel_z.legend()
ax_vel_z.set_title('Z Velocity Comparison')

plt.tight_layout()

# Helper function to interpolate EKF data to PX4 timestamps
def interpolate_to_px4_time(px4_time, ekf_time, ekf_values):
    return np.interp(px4_time, ekf_time, ekf_values)

# Calculate signed error for quaternion components
err_qw = qw_px4 - interpolate_to_px4_time(t_att, time_full, qw_full)
err_qx = qx_px4 - interpolate_to_px4_time(t_att, time_full, qx_full)
err_qy = qy_px4 - interpolate_to_px4_time(t_att, time_full, qy_full)
err_qz = qz_px4 - interpolate_to_px4_time(t_att, time_full, qz_full)

# Calculate signed error for Euler angles (wrap angles to [-180, 180] degrees)
def angle_error_deg(px4_deg, ekf_rad):
    ekf_deg = np.degrees(ekf_rad)
    delta = px4_deg - ekf_deg
    return (delta + 180) % 360 - 180

err_roll = angle_error_deg(np.degrees(roll_px4), interpolate_to_px4_time(t_att, time_full, roll_full))
err_pitch = angle_error_deg(np.degrees(pitch_px4), interpolate_to_px4_time(t_att, time_full, pitch_full))
err_yaw = angle_error_deg(np.degrees(yaw_px4), interpolate_to_px4_time(t_att, time_full, yaw_full))

# Calculate signed error for position
err_x = x_px4 - interpolate_to_px4_time(t, time_full_pos, x_full)
err_y = y_px4 - interpolate_to_px4_time(t, time_full_pos, y_full)
err_z = z_px4 - interpolate_to_px4_time(t, time_full_pos, z_full)

# Calculate signed error for velocity
err_vx = vx_px4 - interpolate_to_px4_time(t, time_full_pos, vx_full)
err_vy = vy_px4 - interpolate_to_px4_time(t, time_full_pos, vy_full)
err_vz = vz_px4 - interpolate_to_px4_time(t, time_full_pos, vz_full)

if WITH_NN:
    # Same for EKF + NN
    err_qw_nn = qw_px4 - interpolate_to_px4_time(t_att, time_nn, qw_nn)
    err_qx_nn = qx_px4 - interpolate_to_px4_time(t_att, time_nn, qx_nn)
    err_qy_nn = qy_px4 - interpolate_to_px4_time(t_att, time_nn, qy_nn)
    err_qz_nn = qz_px4 - interpolate_to_px4_time(t_att, time_nn, qz_nn)

    err_roll_nn = angle_error_deg(np.degrees(roll_px4), interpolate_to_px4_time(t_att, time_nn, roll_nn))
    err_pitch_nn = angle_error_deg(np.degrees(pitch_px4), interpolate_to_px4_time(t_att, time_nn, pitch_nn))
    err_yaw_nn = angle_error_deg(np.degrees(yaw_px4), interpolate_to_px4_time(t_att, time_nn, yaw_nn))

    err_x_nn = x_px4 - interpolate_to_px4_time(t, time_nn_pos, x_nn)
    err_y_nn = y_px4 - interpolate_to_px4_time(t, time_nn_pos, y_nn)
    err_z_nn = z_px4 - interpolate_to_px4_time(t, time_nn_pos, z_nn)

    err_vx_nn = vx_px4 - interpolate_to_px4_time(t, time_nn_pos, vx_nn)
    err_vy_nn = vy_px4 - interpolate_to_px4_time(t, time_nn_pos, vy_nn)
    err_vz_nn = vz_px4 - interpolate_to_px4_time(t, time_nn_pos, vz_nn)


# Plot quaternion error components
fig_err_quat, axs_err_quat = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
axs_err_quat[0].plot(t_att, err_qw, label='EKF', color=colors['EKF Full'])
if WITH_NN: axs_err_quat[0].plot(t_att, err_qw_nn, label='EKF + NN', color=colors['EKF + NN'])
axs_err_quat[0].set_ylabel('Error (qw)')
axs_err_quat[0].legend()
axs_err_quat[0].grid()

axs_err_quat[1].plot(t_att, err_qx, label='EKF', color=colors['EKF Full'])
if WITH_NN: axs_err_quat[1].plot(t_att, err_qx_nn, label='EKF + NN', color=colors['EKF + NN'])
axs_err_quat[1].set_ylabel('Error (qx)')
axs_err_quat[1].legend()
axs_err_quat[1].grid()

axs_err_quat[2].plot(t_att, err_qy, label='EKF', color=colors['EKF Full'])
if WITH_NN: axs_err_quat[2].plot(t_att, err_qy_nn, label='EKF + NN', color=colors['EKF + NN'])
axs_err_quat[2].set_ylabel('Error (qy)')
axs_err_quat[2].legend()
axs_err_quat[2].grid()

axs_err_quat[3].plot(t_att, err_qz, label='EKF', color=colors['EKF Full'])
if WITH_NN: axs_err_quat[3].plot(t_att, err_qz_nn, label='EKF + NN', color=colors['EKF + NN'])
axs_err_quat[3].set_ylabel('Error (qz)')
axs_err_quat[3].set_xlabel('Time [s]')
axs_err_quat[3].legend()
axs_err_quat[3].grid()
fig_err_quat.suptitle('Quaternion Components Signed Error: PX4 - EKF Full')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Plot Euler angle errors
fig_err_euler, axs_err_euler = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
axs_err_euler[0].plot(t_att, err_roll, label='EKF', color=colors['EKF Full'])
if WITH_NN: axs_err_euler[0].plot(t_att, err_roll_nn, label='EKF + NN', color=colors['EKF + NN'])
axs_err_euler[0].set_ylabel('Error Roll [deg]')
axs_err_euler[0].legend()
axs_err_euler[0].grid()

axs_err_euler[1].plot(t_att, err_pitch, label='EKF', color=colors['EKF Full'])
if WITH_NN: axs_err_euler[1].plot(t_att, err_pitch_nn, label='EKF + NN', color=colors['EKF + NN'])
axs_err_euler[1].set_ylabel('Error Pitch [deg]')
axs_err_euler[1].legend()
axs_err_euler[1].grid()

axs_err_euler[2].plot(t_att, err_yaw, label='EKF', color=colors['EKF Full'])
if WITH_NN: axs_err_euler[2].plot(t_att, err_yaw_nn, label='EKF + NN', color=colors['EKF + NN'])
axs_err_euler[2].set_ylabel('Error Yaw [deg]')
axs_err_euler[2].set_xlabel('Time [s]')
axs_err_euler[2].legend()
axs_err_euler[2].grid()
fig_err_euler.suptitle('Euler Angles Signed Error: PX4 - EKF Full')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Plot position errors
fig_err_pos, axs_err_pos = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
axs_err_pos[0].plot(t, err_x, label='EKF', color=colors['EKF Full'])
if WITH_NN: axs_err_pos[0].plot(t, err_x_nn, label='EKF + NN', color=colors['EKF + NN'])
axs_err_pos[0].set_ylabel('Error X [m]')
axs_err_pos[0].legend()
axs_err_pos[0].grid()

axs_err_pos[1].plot(t, err_y, label='EKF', color=colors['EKF Full'])
if WITH_NN: axs_err_pos[1].plot(t, err_y_nn, label='EKF + NN', color=colors['EKF + NN'])
axs_err_pos[1].set_ylabel('Error Y [m]')
axs_err_pos[1].legend()
axs_err_pos[1].grid()

axs_err_pos[2].plot(t, err_z, label='EKF', color=colors['EKF Full'])
if WITH_NN: axs_err_pos[2].plot(t, err_z_nn, label='EKF + NN', color=colors['EKF + NN'])
axs_err_pos[2].set_ylabel('Error Z [m]')
axs_err_pos[2].set_xlabel('Time [s]')
axs_err_pos[2].legend()
axs_err_pos[2].grid()
fig_err_pos.suptitle('Position Signed Error: PX4 - EKF Full')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Plot velocity errors
fig_err_vel, axs_err_vel = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
axs_err_vel[0].plot(t, err_vx, label='EKF', color=colors['EKF Full'])
if WITH_NN: axs_err_vel[0].plot(t, err_vx_nn, label='EKF + NN', color=colors['EKF + NN'])
axs_err_vel[0].set_ylabel('Error Vx [m/s]')
axs_err_vel[0].legend()
axs_err_vel[0].grid()

axs_err_vel[1].plot(t, err_vy, label='EKF', color=colors['EKF Full'])
if WITH_NN: axs_err_vel[1].plot(t, err_vy_nn, label='EKF + NN', color=colors['EKF + NN'])
axs_err_vel[1].set_ylabel('Error Vy [m/s]')
axs_err_vel[1].legend()
axs_err_vel[1].grid()

axs_err_vel[2].plot(t, err_vz, label='EKF', color=colors['EKF Full'])
if WITH_NN: axs_err_vel[2].plot(t, err_vz_nn, label='EKF + NN', color=colors['EKF + NN'])
axs_err_vel[2].set_ylabel('Error Vz [m/s]')
axs_err_vel[2].set_xlabel('Time [s]')
axs_err_vel[2].legend()
axs_err_vel[2].grid()
fig_err_vel.suptitle('Velocity Signed Error: PX4 - EKF Full')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# --- Plot Gyro and Magnetometer Biases from EKF ---
fig_bias, axs_bias = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

# Gyro bias
axs_bias[0].plot(time_full, orientation_full['gyroscope bias (x)'], label='Gyro Bias X')
axs_bias[0].plot(time_full, orientation_full['gyroscope bias (y)'], label='Gyro Bias Y')
axs_bias[0].plot(time_full, orientation_full['gyroscope bias (z)'], label='Gyro Bias Z')
axs_bias[0].set_ylabel('Gyro Bias [rad/s]')
axs_bias[0].legend()
axs_bias[0].grid()
axs_bias[0].set_title('Gyroscope Bias (EKF Full)')

# Magnetometer bias
axs_bias[1].plot(time_full, orientation_full['magnetometer bias (x)'], label='Mag Bias X')
axs_bias[1].plot(time_full, orientation_full['magnetometer bias (y)'], label='Mag Bias Y')
axs_bias[1].plot(time_full, orientation_full['magnetometer bias (z)'], label='Mag Bias Z')
axs_bias[1].set_ylabel('Magnetometer Bias [Gauss]')
axs_bias[1].legend()
axs_bias[1].grid()
axs_bias[1].set_title('Magnetometer Bias (EKF Full)')

axs_bias[1].set_xlabel('Time [s]')

# --- Load covariance data ---
est_cov_full = pd.read_csv(f'./EKF_data/{filename}_orientation_ekf_full_estimation_covariance_diag.csv')
inn_cov_full = pd.read_csv(f'./EKF_data/{filename}_orientation_ekf_full_innovation_covariance_diag.csv')

est_cov_pos_full = pd.read_csv(f'./EKF_data/{filename}_position_ekf_full_estimation_covariance_diag.csv')
inn_cov_pos_full = pd.read_csv(f'./EKF_data/{filename}_position_ekf_full_innovation_covariance_diag.csv')

if WITH_NN:
    est_cov_full_nn = pd.read_csv(f'./EKF_data/{filename}_orientation_ekf_full_nn_estimation_covariance_diag.csv')
    inn_cov_full_nn = pd.read_csv(f'./EKF_data/{filename}_orientation_ekf_full_nn_innovation_covariance_diag.csv')
    est_cov_pos_full_nn = pd.read_csv(f'./EKF_data/{filename}_position_ekf_full_nn_estimation_covariance_diag.csv')
    inn_cov_pos_full_nn = pd.read_csv(f'./EKF_data/{filename}_position_ekf_full_nn_innovation_covariance_diag.csv')

# --- Plot Orientation Estimation Covariance ---
fig_est_cov_orient, axs_est = plt.subplots(2, 2, figsize=(12, 8))
axs_est = axs_est.flatten()

quat_keys = ['quaternion (w)', 'quaternion (x)', 'quaternion (y)', 'quaternion (z)']
for idx, key in enumerate(quat_keys):
    axs_est[idx].plot(time_full, est_cov_full[key], label='EKF Full', color=colors['EKF Full'])
    if WITH_NN:
        axs_est[idx].plot(time_nn, est_cov_full_nn[key], label='EKF + NN', color=colors['EKF + NN'])
    axs_est[idx].set_ylabel(f'P diag [{key}]')
    axs_est[idx].set_xlabel('Time [s]')
    axs_est[idx].grid()
    axs_est[idx].legend()
    axs_est[idx].set_yscale('log')

fig_est_cov_orient.suptitle('Orientation Estimation Covariance (Quaternion States)')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# --- Plot Orientation Bias Estimation Covariance ---
fig_est_cov_bias, axs_est_bias = plt.subplots(2, 3, figsize=(14, 8))
axs_est_bias = axs_est_bias.flatten()

bias_keys = ['gyroscope bias (x)', 'gyroscope bias (y)', 'gyroscope bias (z)',
             'magnetometer bias (x)', 'magnetometer bias (y)', 'magnetometer bias (z)']
for idx, key in enumerate(bias_keys):
    axs_est_bias[idx].plot(time_full, est_cov_full[key], label='EKF Full', color=colors['EKF Full'])
    if WITH_NN:
        axs_est_bias[idx].plot(time_nn, est_cov_full_nn[key], label='EKF + NN', color=colors['EKF + NN'])
    axs_est_bias[idx].set_ylabel(f'P diag [{key}]')
    axs_est_bias[idx].set_xlabel('Time [s]')
    axs_est_bias[idx].grid()
    axs_est_bias[idx].legend()
    axs_est_bias[idx].set_yscale('log')

fig_est_cov_bias.suptitle('Orientation Estimation Covariance (Bias States)')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# --- Plot Orientation Innovation Covariance ---
fig_inn_cov_orient, axs_inn = plt.subplots(2, 3, figsize=(14, 8))
axs_inn = axs_inn.flatten()

meas_keys = ['accelerometer (x)', 'accelerometer (y)', 'accelerometer (z)',
             'magnetometer (x)', 'magnetometer (y)', 'magnetometer (z)']
for idx, key in enumerate(meas_keys):
    axs_inn[idx].plot(time_full, inn_cov_full[key], label='EKF Full', color=colors['EKF Full'])
    if WITH_NN:
        axs_inn[idx].plot(time_nn, inn_cov_full_nn[key], label='EKF + NN', color=colors['EKF + NN'])
    axs_inn[idx].set_ylabel(f'S diag [{key}]')
    axs_inn[idx].set_xlabel('Time [s]')
    axs_inn[idx].grid()
    axs_inn[idx].legend()
    axs_inn[idx].set_yscale('log')

fig_inn_cov_orient.suptitle('Orientation Innovation Covariance (Measurements)')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# --- Plot Position Estimation Covariance ---
fig_est_cov_pos, axs_est_pos = plt.subplots(2, 3, figsize=(14, 8))
axs_est_pos = axs_est_pos.flatten()

pos_state_keys = ['position (x) [m]', 'position (y) [m]', 'position (z) [m]',
                  'velocity (x) [m/s]', 'velocity (y) [m/s]', 'velocity (z) [m/s]']
for idx, key in enumerate(pos_state_keys):
    axs_est_pos[idx].plot(time_full_pos, est_cov_pos_full[key], label='EKF Full', color=colors['EKF Full'])
    if WITH_NN:
        axs_est_pos[idx].plot(time_nn_pos, est_cov_pos_full_nn[key], label='EKF + NN', color=colors['EKF + NN'])
    axs_est_pos[idx].set_ylabel(f'P diag [{key}]')
    axs_est_pos[idx].set_xlabel('Time [s]')
    axs_est_pos[idx].grid()
    axs_est_pos[idx].legend()
    axs_est_pos[idx].set_yscale('log')

fig_est_cov_pos.suptitle('Position Estimation Covariance (States)')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# --- Plot Position Innovation Covariance ---
fig_inn_cov_pos, ax_inn_pos = plt.subplots(figsize=(10, 6))
ax_inn_pos.plot(time_full_pos, inn_cov_pos_full['barometer pressure [Pa]'], 
                label='EKF Full', color=colors['EKF Full'], linewidth=2)
if WITH_NN:
    ax_inn_pos.plot(time_nn_pos, inn_cov_pos_full_nn['barometer pressure [Pa]'], 
                    label='EKF + NN', color=colors['EKF + NN'], linewidth=2)
ax_inn_pos.set_ylabel('S diag [barometer pressure]')
ax_inn_pos.set_xlabel('Time [s]')
ax_inn_pos.grid()
ax_inn_pos.legend()
ax_inn_pos.set_yscale('log')
ax_inn_pos.set_title('Position Innovation Covariance (Barometer Measurement)')

# -- Show all plots ---
plt.tight_layout()
plt.show()
