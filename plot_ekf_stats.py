import os
import numpy as np
import pandas as pd
from pyulog import ULog
from parameters import TEST_FILES
import matplotlib.pyplot as plt

plt.style.use('dark_background')

EKF_DATA_DIR = "./EKF_data"
PX4_DATA_DIR = "./Ishaan_TakeHomeData_3Flights"
PLOTS_DIR = "./plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

def load_ekf_data(file_path):
    df = pd.read_csv(file_path)
    # Assumes columns: time, position (x) [m], position (y) [m], position (z) [m]
    return (
        df['time'].values, df['position (x) [m]'].values, 
        df['position (y) [m]'].values, df['position (z) [m]'].values,
        df['velocity (x) [m/s]'].values, 
        df['velocity (y) [m/s]'].values, df['velocity (z) [m/s]'].values
    )

def load_px4_data(ulog_path):
    log = ULog(ulog_path)
    pos_topic = next(x for x in log.data_list if x.name == 'vehicle_local_position')
    t = pos_topic.data['timestamp'] * 1e-6
    x = pos_topic.data['x'] - pos_topic.data['x'][0]
    y = pos_topic.data['y'] - pos_topic.data['y'][0]
    z = pos_topic.data['z'] - pos_topic.data['z'][0]
    vel_x = pos_topic.data['vx']
    vel_y = pos_topic.data['vy']
    vel_z = pos_topic.data['vz']
    return t, x, y, z, vel_x, vel_y, vel_z

def interpolate_to_px4_time(px4_time, ekf_time, ekf_values):
    return np.interp(px4_time, ekf_time, ekf_values)

def plot_error_stats(errors, file_name, tick_labels):
    plt.figure(figsize=(10, 6))
    plt.title(f"EKF Error Stats Distribution: {file_name}")
    plt.boxplot(errors, tick_labels=tick_labels, showfliers=False)
    plt.ylabel('Error (PX4 - EKF)')
    plt.grid(True)
    plt.savefig(os.path.join(PLOTS_DIR, f"ekf_error_stats_{file_name}.png"))
    plt.close()

def plot_single_error_box(error, file_name, label):
    plt.figure(figsize=(6, 6))
    plt.title(f"EKF Error Stats: {file_name} ({label})")
    plt.boxplot(error, tick_labels=[label], showfliers=False)
    plt.ylabel('Error (PX4 - EKF)')
    plt.grid(True)
    plt.savefig(os.path.join(PLOTS_DIR, f"ekf_error_stats_{file_name}_{label}.png"))
    plt.close()

def plot_error_histogram(error, file_name, label, bins=50):
    plt.figure(figsize=(8, 6))
    plt.title(f"EKF Error Histogram: {file_name} ({label})")
    plt.hist(error, bins=bins, color='blue', alpha=0.7)
    plt.xlabel('Error (PX4 - EKF)')
    plt.ylabel('Number of Occurrences')
    plt.grid(True)
    plt.savefig(os.path.join(PLOTS_DIR, f"ekf_error_hist_{file_name}_{label}.png"))
    plt.close()

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

def angle_error_deg(px4_deg, ekf_rad):
    ekf_deg = np.degrees(ekf_rad)
    delta = px4_deg - ekf_deg
    return (delta + 180) % 360 - 180

def wrap_angle_deg(angle):
    """Wrap angle to [-180, 180] degrees."""
    return (angle + 180) % 360 - 180

def calc_rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))

# Initialize lists to collect errors across all files
all_errors = {'x': [], 'y': [], 'z': []}
all_vel_errors = {'vx': [], 'vy': [], 'vz': []}
all_nn_vel_errors = {'vx': [], 'vy': [], 'vz': []}
all_quat_errors = {'qw': [], 'qx': [], 'qy': [], 'qz': []}
all_euler_errors = {'roll': [], 'pitch': [], 'yaw': []}
all_nn_errors = {'x': [], 'y': [], 'z': []}
all_nn_quat_errors = {'qw': [], 'qx': [], 'qy': [], 'qz': []}
all_nn_euler_errors = {'roll': [], 'pitch': [], 'yaw': []}
rmse_errors = {'x': [], 'y': [], 'z': []}
rmse_quat_errors = {'qw': [], 'qx': [], 'qy': [], 'qz': []}
rmse_euler_errors = {'roll': [], 'pitch': [], 'yaw': []}
rmse_nn_errors = {'x': [], 'y': [], 'z': []}
rmse_nn_quat_errors = {'qw': [], 'qx': [], 'qy': [], 'qz': []}
rmse_nn_euler_errors = {'roll': [], 'pitch': [], 'yaw': []}

for test_file in TEST_FILES:
    ekf_path = os.path.join(
        EKF_DATA_DIR, f"{os.path.splitext(test_file)[0]}_position_ekf_full_states.csv"
    )
    ekf_orient_path = os.path.join(
        EKF_DATA_DIR, f"{os.path.splitext(test_file)[0]}_orientation_ekf_full_states.csv"
    )
    px4_path = os.path.join(PX4_DATA_DIR, f"{test_file}.ulg")
    if not (
        (os.path.exists(ekf_path) and os.path.exists(px4_path) and 
         os.path.exists(ekf_orient_path))
    ):
        print(f"Missing data for {test_file}, skipping.")
        continue

    ekf_time, ekf_x, ekf_y, ekf_z, ekf_vel_x , ekf_vel_y, ekf_vel_z = (
        load_ekf_data(ekf_path)
    )
    px4_time, px4_x, px4_y, px4_z, px4_vel_x, px4_vel_y, px4_vel_z = (
        load_px4_data(px4_path)
    )

    # Interpolate EKF to PX4 timestamps
    ekf_x_interp = interpolate_to_px4_time(px4_time, ekf_time, ekf_x)
    ekf_y_interp = interpolate_to_px4_time(px4_time, ekf_time, ekf_y)
    ekf_z_interp = interpolate_to_px4_time(px4_time, ekf_time, ekf_z)

    error_x = px4_x - ekf_x_interp
    error_y = px4_y - ekf_y_interp
    error_z = px4_z - ekf_z_interp

    # --- Quaternion error stats ---
    ekf_orient_df = pd.read_csv(ekf_orient_path)
    qw_ekf = ekf_orient_df['quaternion (w)'].values
    qx_ekf = ekf_orient_df['quaternion (x)'].values
    qy_ekf = ekf_orient_df['quaternion (y)'].values
    qz_ekf = ekf_orient_df['quaternion (z)'].values
    ekf_orient_time = ekf_orient_df['time'].values

    log = ULog(px4_path)
    att_topic = next(x for x in log.data_list if x.name == 'vehicle_attitude')
    t_att = att_topic.data['timestamp'] * 1e-6
    qw_px4 = att_topic.data['q[0]']
    qx_px4 = att_topic.data['q[1]']
    qy_px4 = att_topic.data['q[2]']
    qz_px4 = att_topic.data['q[3]']

    qw_ekf_interp = interpolate_to_px4_time(t_att, ekf_orient_time, qw_ekf)
    qx_ekf_interp = interpolate_to_px4_time(t_att, ekf_orient_time, qx_ekf)
    qy_ekf_interp = interpolate_to_px4_time(t_att, ekf_orient_time, qy_ekf)
    qz_ekf_interp = interpolate_to_px4_time(t_att, ekf_orient_time, qz_ekf)

    error_qw = qw_px4 - qw_ekf_interp
    error_qx = qx_px4 - qx_ekf_interp
    error_qy = qy_px4 - qy_ekf_interp
    error_qz = qz_px4 - qz_ekf_interp

    # --- Euler angle error stats ---
    roll_px4, pitch_px4, yaw_px4 = quaternion_to_euler(qw_px4, qx_px4, qy_px4, qz_px4)
    roll_ekf, pitch_ekf, yaw_ekf = quaternion_to_euler(qw_ekf_interp, qx_ekf_interp, qy_ekf_interp, qz_ekf_interp)

    error_roll = angle_error_deg(np.degrees(roll_px4), roll_ekf)
    error_pitch = angle_error_deg(np.degrees(pitch_px4), pitch_ekf)
    error_yaw = angle_error_deg(np.degrees(yaw_px4), yaw_ekf)

    # Collect errors for combined stats
    all_errors['x'].append(error_x)
    all_errors['y'].append(error_y)
    all_errors['z'].append(error_z)

    all_quat_errors['qw'].append(error_qw)
    all_quat_errors['qx'].append(error_qx)
    all_quat_errors['qy'].append(error_qy)
    all_quat_errors['qz'].append(error_qz)

    all_euler_errors['roll'].append(error_roll)
    all_euler_errors['pitch'].append(error_pitch)
    all_euler_errors['yaw'].append(error_yaw)

    # --- EKF+NN Position ---
    ekf_nn_path = os.path.join(EKF_DATA_DIR, f"{os.path.splitext(test_file)[0]}_position_ekf_full_nn_states.csv")
    ekf_orient_nn_path = os.path.join(EKF_DATA_DIR, f"{os.path.splitext(test_file)[0]}_orientation_ekf_full_nn_states.csv")
    if not (os.path.exists(ekf_nn_path) and os.path.exists(ekf_orient_nn_path)):
        print(f"Missing EKF+NN data for {test_file}, skipping NN.")
        continue

    # Position
    ekf_nn_df = pd.read_csv(ekf_nn_path)
    ekf_nn_time = ekf_nn_df['time'].values
    ekf_nn_x = ekf_nn_df['position (x) [m]'].values
    ekf_nn_y = ekf_nn_df['position (y) [m]'].values
    ekf_nn_z = ekf_nn_df['position (z) [m]'].values

    ekf_nn_x_interp = interpolate_to_px4_time(px4_time, ekf_nn_time, ekf_nn_x)
    ekf_nn_y_interp = interpolate_to_px4_time(px4_time, ekf_nn_time, ekf_nn_y)
    ekf_nn_z_interp = interpolate_to_px4_time(px4_time, ekf_nn_time, ekf_nn_z)

    nn_error_x = px4_x - ekf_nn_x_interp
    nn_error_y = px4_y - ekf_nn_y_interp
    nn_error_z = px4_z - ekf_nn_z_interp

    all_nn_errors['x'].append(nn_error_x)
    all_nn_errors['y'].append(nn_error_y)
    all_nn_errors['z'].append(nn_error_z)

    # Orientation
    ekf_orient_nn_df = pd.read_csv(ekf_orient_nn_path)
    nn_qw = ekf_orient_nn_df['quaternion (w)'].values
    nn_qx = ekf_orient_nn_df['quaternion (x)'].values
    nn_qy = ekf_orient_nn_df['quaternion (y)'].values
    nn_qz = ekf_orient_nn_df['quaternion (z)'].values
    nn_orient_time = ekf_orient_nn_df['time'].values

    nn_qw_interp = interpolate_to_px4_time(t_att, nn_orient_time, nn_qw)
    nn_qx_interp = interpolate_to_px4_time(t_att, nn_orient_time, nn_qx)
    nn_qy_interp = interpolate_to_px4_time(t_att, nn_orient_time, nn_qy)
    nn_qz_interp = interpolate_to_px4_time(t_att, nn_orient_time, nn_qz)

    nn_error_qw = qw_px4 - nn_qw_interp
    nn_error_qx = qx_px4 - nn_qx_interp
    nn_error_qy = qy_px4 - nn_qy_interp
    nn_error_qz = qz_px4 - nn_qz_interp

    all_nn_quat_errors['qw'].append(nn_error_qw)
    all_nn_quat_errors['qx'].append(nn_error_qx)
    all_nn_quat_errors['qy'].append(nn_error_qy)
    all_nn_quat_errors['qz'].append(nn_error_qz)

    nn_roll, nn_pitch, nn_yaw = quaternion_to_euler(nn_qw_interp, nn_qx_interp, nn_qy_interp, nn_qz_interp)
    nn_error_roll = angle_error_deg(np.degrees(roll_px4), nn_roll)
    nn_error_pitch = angle_error_deg(np.degrees(pitch_px4), nn_pitch)
    nn_error_yaw = wrap_angle_deg(np.degrees(yaw_px4) - np.degrees(nn_yaw))

    all_nn_euler_errors['roll'].append(nn_error_roll)
    all_nn_euler_errors['pitch'].append(nn_error_pitch)
    all_nn_euler_errors['yaw'].append(nn_error_yaw)

    # Interpolate EKF velocity to PX4 timestamps
    ekf_vel_x_interp = interpolate_to_px4_time(px4_time, ekf_time, ekf_vel_x)
    ekf_vel_y_interp = interpolate_to_px4_time(px4_time, ekf_time, ekf_vel_y)
    ekf_vel_z_interp = interpolate_to_px4_time(px4_time, ekf_time, ekf_vel_z)

    error_vx = px4_vel_x - ekf_vel_x_interp
    error_vy = px4_vel_y - ekf_vel_y_interp
    error_vz = px4_vel_z - ekf_vel_z_interp

    all_vel_errors['vx'].append(error_vx)
    all_vel_errors['vy'].append(error_vy)
    all_vel_errors['vz'].append(error_vz)

    # --- EKF+NN velocity ---
    nn_vel_x = ekf_nn_df['velocity (x) [m/s]'].values
    nn_vel_y = ekf_nn_df['velocity (y) [m/s]'].values
    nn_vel_z = ekf_nn_df['velocity (z) [m/s]'].values
    nn_vel_time = ekf_nn_df['time'].values

    nn_vel_x_interp = interpolate_to_px4_time(px4_time, nn_vel_time, nn_vel_x)
    nn_vel_y_interp = interpolate_to_px4_time(px4_time, nn_vel_time, nn_vel_y)
    nn_vel_z_interp = interpolate_to_px4_time(px4_time, nn_vel_time, nn_vel_z)

    nn_error_vx = px4_vel_x - nn_vel_x_interp
    nn_error_vy = px4_vel_y - nn_vel_y_interp
    nn_error_vz = px4_vel_z - nn_vel_z_interp

    all_nn_vel_errors['vx'].append(nn_error_vx)
    all_nn_vel_errors['vy'].append(nn_error_vy)
    all_nn_vel_errors['vz'].append(nn_error_vz)

# After processing all test files, combine and plot overall stats
if all_errors['x'] and all_errors['y'] and all_errors['z']:
    combined_errors = [np.concatenate(all_errors[k]) for k in ['x', 'y', 'z']]
    combined_quat_errors = [np.concatenate(all_quat_errors[k]) for k in ['qw', 'qx', 'qy', 'qz']]
    combined_euler_errors = [np.concatenate(all_euler_errors[k]) for k in ['roll', 'pitch', 'yaw']]

    # Plot combined stats
    plot_error_stats(combined_errors, "combined_pos", ['x', 'y', 'z'])
    plot_error_stats(combined_quat_errors, "combined_quat", ['qw', 'qx', 'qy', 'qz'])
    plot_error_stats(combined_euler_errors, "combined_euler", ['roll', 'pitch', 'yaw'])

    # Plot single box and histogram for each state
    for err, lbl in zip(combined_errors, ['x', 'y', 'z']):
        plot_single_error_box(err, "combined_pos", lbl)
        plot_error_histogram(err, "combined_pos", lbl)

    for err, lbl in zip(combined_quat_errors, ['qw', 'qx', 'qy', 'qz']):
        plot_single_error_box(err, "combined_quat", lbl)
        plot_error_histogram(err, "combined_quat", lbl)

    for err, lbl in zip(combined_euler_errors, ['roll', 'pitch', 'yaw']):
        plot_single_error_box(err, "combined_euler", lbl)
        plot_error_histogram(err, "combined_euler", lbl)

    print("Combined EKF error stats plots saved.")
else:
    print("No error data collected. Combined plots not generated.")

# After processing all test files, combine and plot overall stats for EKF+NN
if all_nn_errors['x'] and all_nn_errors['y'] and all_nn_errors['z']:
    combined_nn_errors = [np.concatenate(all_nn_errors[k]) for k in ['x', 'y', 'z']]
    combined_nn_quat_errors = [np.concatenate(all_nn_quat_errors[k]) for k in ['qw', 'qx', 'qy', 'qz']]
    combined_nn_euler_errors = [np.concatenate(all_nn_euler_errors[k]) for k in ['roll', 'pitch', 'yaw']]

    # Plot combined stats for EKF+NN
    plot_error_stats(combined_nn_errors, "combined_pos_nn", ['x', 'y', 'z'])
    plot_error_stats(combined_nn_quat_errors, "combined_quat_nn", ['qw', 'qx', 'qy', 'qz'])
    plot_error_stats(combined_nn_euler_errors, "combined_euler_nn", ['roll', 'pitch', 'yaw'])

    # Plot single box and histogram for each state
    for err, lbl in zip(combined_nn_errors, ['x', 'y', 'z']):
        plot_single_error_box(err, "combined_pos_nn", lbl)
        plot_error_histogram(err, "combined_pos_nn", lbl)

    for err, lbl in zip(combined_nn_quat_errors, ['qw', 'qx', 'qy', 'qz']):
        plot_single_error_box(err, "combined_quat_nn", lbl)
        plot_error_histogram(err, "combined_quat_nn", lbl)

    for err, lbl in zip(combined_nn_euler_errors, ['roll', 'pitch', 'yaw']):
        plot_single_error_box(err, "combined_euler_nn", lbl)
        plot_error_histogram(err, "combined_euler_nn", lbl)

    print("Combined EKF+NN error stats plots saved.")
else:
    print("No EKF+NN error data collected. Combined NN plots not generated.")

def plot_overlay_boxplots(errors_ekf, errors_nn, file_name, tick_labels):
    plt.figure(figsize=(10, 6))
    plt.title(f"EKF vs EKF+NN Error Stats: {file_name}")
    boxprops_ekf = dict(color='blue')
    boxprops_nn = dict(color='orange')
    # Plot EKF
    bp1 = plt.boxplot(errors_ekf, positions=np.arange(len(errors_ekf))-0.2, widths=0.3,
                      patch_artist=True, boxprops=boxprops_ekf, medianprops=dict(color='blue'), showfliers=False)
    # Plot EKF+NN
    bp2 = plt.boxplot(errors_nn, positions=np.arange(len(errors_nn))+0.2, widths=0.3,
                      patch_artist=True, boxprops=boxprops_nn, medianprops=dict(color='orange'), showfliers=False)
    plt.xticks(np.arange(len(tick_labels)), tick_labels)
    plt.ylabel('Error (PX4 - EKF)')
    plt.legend([bp1["boxes"][0], bp2["boxes"][0]], ["EKF", "EKF+NN"], loc="best")
    plt.grid(True)
    plt.savefig(os.path.join(PLOTS_DIR, f"overlay_boxplot_{file_name}.png"))
    plt.close()

# After combining errors, call this for each group:
if all_errors['x'] and all_nn_errors['x']:
    plot_overlay_boxplots(combined_errors, combined_nn_errors, "pos", ['x', 'y', 'z'])
    plot_overlay_boxplots(combined_quat_errors, combined_nn_quat_errors, "quat", ['qw', 'qx', 'qy', 'qz'])
    plot_overlay_boxplots(combined_euler_errors, combined_nn_euler_errors, "euler", ['roll', 'pitch', 'yaw'])

def plot_overlay_boxplot_pz(error_ekf, error_nn, file_name):
    plt.figure(figsize=(6, 6))
    plt.title(f"EKF vs EKF+NN Error Stats: {file_name} (z)")
    boxprops_ekf = dict(color='blue')
    boxprops_nn = dict(color='orange')
    # Plot EKF
    bp1 = plt.boxplot([error_ekf], positions=[0.8], widths=0.3,
                      patch_artist=True, boxprops=boxprops_ekf, medianprops=dict(color='blue'), showfliers=False)
    # Plot EKF+NN
    bp2 = plt.boxplot([error_nn], positions=[1.2], widths=0.3,
                      patch_artist=True, boxprops=boxprops_nn, medianprops=dict(color='orange'), showfliers=False)
    plt.xticks([0.8, 1.2], ["EKF", "EKF+NN"])
    plt.ylabel('Error (PX4 - EKF)')
    plt.grid(True)
    plt.savefig(os.path.join(PLOTS_DIR, f"overlay_boxplot_{file_name}_z.png"))
    plt.close()

# After combining errors, call this:
if all_errors['z'] and all_nn_errors['z']:
    plot_overlay_boxplot_pz(combined_errors[2], combined_nn_errors[2], "pos")

def calc_rmse_from_list(error_list):
    arr = np.concatenate(error_list)
    return np.sqrt(np.mean(arr ** 2))

if all_errors['x'] and all_errors['y'] and all_errors['z']:
    rmse_x = calc_rmse_from_list(all_errors['x'])
    rmse_y = calc_rmse_from_list(all_errors['y'])
    rmse_z = calc_rmse_from_list(all_errors['z'])
    print(f"Combined EKF Position RMSE: x={rmse_x:.3f}, y={rmse_y:.3f}, z={rmse_z:.3f}")

    rmse_qw = calc_rmse_from_list(all_quat_errors['qw'])
    rmse_qx = calc_rmse_from_list(all_quat_errors['qx'])
    rmse_qy = calc_rmse_from_list(all_quat_errors['qy'])
    rmse_qz = calc_rmse_from_list(all_quat_errors['qz'])
    print(f"Combined EKF Quaternion RMSE: qw={rmse_qw:.3f}, qx={rmse_qx:.3f}, qy={rmse_qy:.3f}, qz={rmse_qz:.3f}")

    rmse_roll = calc_rmse_from_list(all_euler_errors['roll'])
    rmse_pitch = calc_rmse_from_list(all_euler_errors['pitch'])
    rmse_yaw = calc_rmse_from_list(all_euler_errors['yaw'])
    print(f"Combined EKF Euler RMSE: roll={rmse_roll:.3f}, pitch={rmse_pitch:.3f}, yaw={rmse_yaw:.3f}")

if all_nn_errors['x'] and all_nn_errors['y'] and all_nn_errors['z']:
    nn_rmse_x = calc_rmse_from_list(all_nn_errors['x'])
    nn_rmse_y = calc_rmse_from_list(all_nn_errors['y'])
    nn_rmse_z = calc_rmse_from_list(all_nn_errors['z'])
    print(f"Combined EKF+NN Position RMSE: x={nn_rmse_x:.3f}, y={nn_rmse_y:.3f}, z={nn_rmse_z:.3f}")

    nn_rmse_qw = calc_rmse_from_list(all_nn_quat_errors['qw'])
    nn_rmse_qx = calc_rmse_from_list(all_nn_quat_errors['qx'])
    nn_rmse_qy = calc_rmse_from_list(all_nn_quat_errors['qy'])
    nn_rmse_qz = calc_rmse_from_list(all_nn_quat_errors['qz'])
    print(f"Combined EKF+NN Quaternion RMSE: qw={nn_rmse_qw:.3f}, qx={nn_rmse_qx:.3f}, qy={nn_rmse_qy:.3f}, qz={nn_rmse_qz:.3f}")

    nn_rmse_roll = calc_rmse_from_list(all_nn_euler_errors['roll'])
    nn_rmse_pitch = calc_rmse_from_list(all_nn_euler_errors['pitch'])
    nn_rmse_yaw = calc_rmse_from_list(all_nn_euler_errors['yaw'])
    print(f"Combined EKF+NN Euler RMSE: roll={nn_rmse_roll:.3f}, pitch={nn_rmse_pitch:.3f}, yaw={nn_rmse_yaw:.3f}")

if all_vel_errors['vx'] and all_vel_errors['vy'] and all_vel_errors['vz']:
    rmse_vx = calc_rmse_from_list(all_vel_errors['vx'])
    rmse_vy = calc_rmse_from_list(all_vel_errors['vy'])
    rmse_vz = calc_rmse_from_list(all_vel_errors['vz'])
    print(f"Combined EKF Velocity RMSE: vx={rmse_vx:.3f}, vy={rmse_vy:.3f}, vz={rmse_vz:.3f}")

if all_nn_vel_errors['vx'] and all_nn_vel_errors['vy'] and all_nn_vel_errors['vz']:
    nn_rmse_vx = calc_rmse_from_list(all_nn_vel_errors['vx'])
    nn_rmse_vy = calc_rmse_from_list(all_nn_vel_errors['vy'])
    nn_rmse_vz = calc_rmse_from_list(all_nn_vel_errors['vz'])
    print(f"Combined EKF+NN Velocity RMSE: vx={nn_rmse_vx:.3f}, vy={nn_rmse_vy:.3f}, vz={nn_rmse_vz:.3f}")
