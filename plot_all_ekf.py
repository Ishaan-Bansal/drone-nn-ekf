import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pyulog import ULog
from parameters import TEST_FILES, TRAINING_FILES

plt.style.use('dark_background')

EKF_DATA_DIR = "./ekf_data"
PX4_DATA_DIR = "./flight_data"

def load_ekf_data(file_path):
    df = pd.read_csv(file_path)
    return df['time'].values, df['position (x) [m]'].values, df['position (y) [m]'].values, df['position (z) [m]'].values

def load_px4_data(ulog_path):
    log = ULog(ulog_path)
    pos_topic = next(x for x in log.data_list if x.name == 'vehicle_local_position')
    t = pos_topic.data['timestamp'] * 1e-6
    x = pos_topic.data['x'] - pos_topic.data['x'][0]
    y = pos_topic.data['y'] - pos_topic.data['y'][0]
    z = pos_topic.data['z'] - pos_topic.data['z'][0]
    att_topic = next(x for x in log.data_list if x.name == 'vehicle_attitude')
    t_att = att_topic.data['timestamp'] * 1e-6
    qw = att_topic.data['q[0]']
    qx = att_topic.data['q[1]']
    qy = att_topic.data['q[2]']
    qz = att_topic.data['q[3]']
    return t, x, y, z, t_att, qw, qx, qy, qz

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

def interpolate_to_px4_time(px4_time, ekf_time, ekf_values):
    return np.interp(px4_time, ekf_time, ekf_values)

# Prepare figures for each state
figs = {}
axes = {}
states = ['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz', 'roll', 'pitch', 'yaw']
for state in states:
    figs[state], axes[state] = plt.subplots(figsize=(10, 6))
    axes[state].set_title(f"{state.upper()} Comparison Across All Flights")
    axes[state].set_xlabel("Time [s]")
    axes[state].set_ylabel(state)

all_files = TEST_FILES + TRAINING_FILES
print("Plotting data for files:", all_files)
cmap = plt.get_cmap('turbo', len(all_files))
file_colors = {test_file: cmap(i) for i, test_file in enumerate(all_files)}

for test_file in all_files:
    ekf_path = os.path.join(EKF_DATA_DIR, f"{os.path.splitext(test_file)[0]}_position_ekf_full_states.csv")
    ekf_orient_path = os.path.join(EKF_DATA_DIR, f"{os.path.splitext(test_file)[0]}_orientation_ekf_full_states.csv")
    px4_path = os.path.join(PX4_DATA_DIR, f"{test_file}.ulg")
    if not (os.path.exists(ekf_path) and os.path.exists(px4_path) and os.path.exists(ekf_orient_path)):
        print(f"Missing data for {test_file}, skipping.")
        continue

    # Load EKF position and orientation
    ekf_time, ekf_x, ekf_y, ekf_z = load_ekf_data(ekf_path)
    ekf_time -= ekf_time[0]  # Normalize EKF time to start at 0
    ekf_orient_df = pd.read_csv(ekf_orient_path)
    ekf_orient_time = ekf_orient_df['time'].values
    ekf_orient_time -= ekf_orient_time[0]  # Normalize time to start at 0
    qw_ekf = ekf_orient_df['quaternion (w)'].values
    qx_ekf = ekf_orient_df['quaternion (x)'].values
    qy_ekf = ekf_orient_df['quaternion (y)'].values
    qz_ekf = ekf_orient_df['quaternion (z)'].values
    roll_ekf, pitch_ekf, yaw_ekf = quaternion_to_euler(qw_ekf, qx_ekf, qy_ekf, qz_ekf)

    # Load PX4 data
    t_px4, x_px4, y_px4, z_px4, t_att, qw_px4, qx_px4, qy_px4, qz_px4 = load_px4_data(px4_path)
    t_att -= t_att[0]  # Normalize PX4 time to start at 0
    t_px4 -= t_px4[0]  # Normalize PX4 time to start at 0
    roll_px4, pitch_px4, yaw_px4 = quaternion_to_euler(qw_px4, qx_px4, qy_px4, qz_px4)
    yaw_px4 -= yaw_px4[0]  # Normalize yaw to start at 0

    # Interpolate EKF to PX4 timestamps
    ekf_x_interp = interpolate_to_px4_time(t_px4, ekf_time, ekf_x)
    ekf_y_interp = interpolate_to_px4_time(t_px4, ekf_time, ekf_y)
    ekf_z_interp = interpolate_to_px4_time(t_px4, ekf_time, ekf_z)
    qw_ekf_interp = interpolate_to_px4_time(t_att, ekf_orient_time, qw_ekf)
    qx_ekf_interp = interpolate_to_px4_time(t_att, ekf_orient_time, qx_ekf)
    qy_ekf_interp = interpolate_to_px4_time(t_att, ekf_orient_time, qy_ekf)
    qz_ekf_interp = interpolate_to_px4_time(t_att, ekf_orient_time, qz_ekf)
    roll_ekf_interp = interpolate_to_px4_time(t_att, ekf_orient_time, roll_ekf)
    pitch_ekf_interp = interpolate_to_px4_time(t_att, ekf_orient_time, pitch_ekf)
    yaw_ekf_interp = interpolate_to_px4_time(t_att, ekf_orient_time, yaw_ekf)

    color = file_colors[test_file]

    # Plot each state for this flight
    axes['x'].plot(t_px4, x_px4, label=f'PX4 {test_file}', alpha=0.5, color=color)
    axes['x'].plot(t_px4, ekf_x_interp, label=f'EKF {test_file}', linestyle='--', alpha=0.7, color=color)
    axes['y'].plot(t_px4, y_px4, label=f'PX4 {test_file}', alpha=0.5, color=color)
    axes['y'].plot(t_px4, ekf_y_interp, label=f'EKF {test_file}', linestyle='--', alpha=0.7, color=color)
    axes['z'].plot(t_px4, z_px4, label=f'PX4 {test_file}', alpha=0.5, color=color)
    axes['z'].plot(t_px4, ekf_z_interp, label=f'EKF {test_file}', linestyle='--', alpha=0.7, color=color)

    axes['qw'].plot(t_att, qw_px4, label=f'PX4 {test_file}', alpha=0.5, color=color)
    axes['qw'].plot(t_att, qw_ekf_interp, label=f'EKF {test_file}', linestyle='--', alpha=0.7, color=color)
    axes['qx'].plot(t_att, qx_px4, label=f'PX4 {test_file}', alpha=0.5, color=color)
    axes['qx'].plot(t_att, qx_ekf_interp, label=f'EKF {test_file}', linestyle='--', alpha=0.7, color=color)
    axes['qy'].plot(t_att, qy_px4, label=f'PX4 {test_file}', alpha=0.5, color=color)
    axes['qy'].plot(t_att, qy_ekf_interp, label=f'EKF {test_file}', linestyle='--', alpha=0.7, color=color)
    axes['qz'].plot(t_att, qz_px4, label=f'PX4 {test_file}', alpha=0.5, color=color)
    axes['qz'].plot(t_att, qz_ekf_interp, label=f'EKF {test_file}', linestyle='--', alpha=0.7, color=color)

    axes['roll'].plot(t_att, np.degrees(roll_px4), label=f'PX4 {test_file}', alpha=0.5, color=color)
    axes['roll'].plot(t_att, np.degrees(roll_ekf_interp), label=f'EKF {test_file}', linestyle='--', alpha=0.7, color=color)
    axes['pitch'].plot(t_att, np.degrees(pitch_px4), label=f'PX4 {test_file}', alpha=0.5, color=color)
    axes['pitch'].plot(t_att, np.degrees(pitch_ekf_interp), label=f'EKF {test_file}', linestyle='--', alpha=0.7, color=color)
    axes['yaw'].plot(t_att, np.degrees(yaw_px4), label=f'PX4 {test_file}', alpha=0.5, color=color)
    axes['yaw'].plot(t_att, np.degrees(yaw_ekf_interp), label=f'EKF {test_file}', linestyle='--', alpha=0.7, color=color)

# Finalize and save figures
for state in states:
    axes[state].legend(loc='best', fontsize='small', ncol=2)
    axes[state].grid(True)
    figs[state].tight_layout()
    figs[state].savefig(f"./plots/all_flights_{state}.png")

plt.show()