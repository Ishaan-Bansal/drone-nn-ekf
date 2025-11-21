import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pyulog import ULog
from sensors import Sensor
from parameters import (
    TRAINING_FILES, TEST_FILES, 
    NUM_EPOCHS, NUM_NEURONS, NUM_LAYERS, BATCH_SIZE, LEARNING_RATE,
    ACCEL_LPF_ALPHA_X, ACCEL_LPF_ALPHA_Y, ACCEL_LPF_ALPHA_Z,
    MAG_LPF_ALPHA, GYRO_LPF_ALPHA, BARO_LPF_ALPHA
)
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def smooth_targets(targets, window=10):
    """
    Smooth each output dimension with a moving average window.
    
    :param targets: numpy array of shape (N_samples, num_outputs)
    :param window: size of the moving average window (samples)
    :return: smoothed_targets with the same shape as input
    """
    smoothed = np.empty_like(targets)
    for i in range(targets.shape[1]):
        smoothed[:, i] = np.convolve(targets[:, i], np.ones(window)/window, mode='same')
    return smoothed


class EKFResidualDataset(Dataset):
    def __init__(self, ekf_triplets,  # list of (orientation_csv, position_csv, px4_ulog)
                 px4_state_topic='vehicle_local_position',
                 px4_attitude_topic='vehicle_attitude',
                 accel_topic='sensor_combined',
                 gyro_topic='sensor_combined',
                 baro_topic='sensor_baro',
                 mag_topic='sensor_mag',
                 time_tolerance=0.02,
                 seq_len=1):
        """
        ekf_triplets: list of tuples
            (orientation_csv_path, position_csv_path, px4_ulog_path)
        """

        all_inputs, all_targets, all_times = [], [], []

        for ori_csv, pos_csv, ulog_file in ekf_triplets:
            # --- Load EKF position+velocity CSV ---
            pos_df = pd.read_csv(pos_csv)
            time_pos = pos_df['time'].values
            pos_keys = [
                'position (x) [m]', 'position (y) [m]', 'position (z) [m]',
                'velocity (x) [m/s]', 'velocity (y) [m/s]', 'velocity (z) [m/s]'
            ]
            pos_states = pos_df[pos_keys].values

            # --- Load EKF orientation CSV ---
            ori_df = pd.read_csv(ori_csv)
            time_ori = ori_df['time'].values
            ori_keys = ['quaternion (w)', 'quaternion (x)', 'quaternion (y)', 'quaternion (z)']
            ori_states = ori_df[ori_keys].values

            # --- Interpolate orientation to match position timestamps ---
            ori_interp = np.empty((len(time_pos), 4), dtype=np.float32)
            for i in range(4):
                ori_interp[:, i] = np.interp(time_pos, time_ori, ori_states[:, i])

            # --- Unified EKF state vector ---
            ekf_states = np.hstack([pos_states, ori_interp])  # shape (N, 10)
            ekf_time = time_pos

            # --- Load PX4 Log ---
            log = ULog(ulog_file)

            # PX4 reference state
            posvel_topic = next(x for x in log.data_list if x.name == px4_state_topic)
            att_topic = next(x for x in log.data_list if x.name == px4_attitude_topic)

            posvel_time = posvel_topic.data['timestamp'] * 1e-6
            pos_data = np.vstack([posvel_topic.data['x'],
                                posvel_topic.data['y'],
                                posvel_topic.data['z']]).T
            pos_data -= pos_data[0]
            vel_data = np.vstack([posvel_topic.data['vx'],
                                posvel_topic.data['vy'],
                                posvel_topic.data['vz']]).T
            vel_data -= vel_data[0]

            att_time = att_topic.data['timestamp'] * 1e-6
            quat_data = np.vstack([att_topic.data['q[0]'],
                                att_topic.data['q[1]'],
                                att_topic.data['q[2]'],
                                att_topic.data['q[3]']]).T

            # Align attitude to position/velocity timestamps
            quat_data_synced = np.empty((len(posvel_time), 4))
            for i, t in enumerate(posvel_time):
                att_idx = np.argmin(np.abs(att_time - t))
                quat_data_synced[i, :] = quat_data[att_idx]

            px4_state_time = posvel_time
            px4_states = np.hstack([pos_data, vel_data, quat_data_synced])

            # --- PX4 Sensors using Sensor class ---
            sensor_combined = next(x for x in log.data_list if x.name == accel_topic)
            mag_topic_obj = next(x for x in log.data_list if x.name == mag_topic)
            baro_topic_obj = next(x for x in log.data_list if x.name == baro_topic)

            # Create Sensor instances
            accel_sensor = Sensor(
                'accelerometer', sensor_combined,
                ['accelerometer_m_s2[0]', 'accelerometer_m_s2[1]', 'accelerometer_m_s2[2]'],
                alpha=np.array([ACCEL_LPF_ALPHA_X, ACCEL_LPF_ALPHA_Y, ACCEL_LPF_ALPHA_Z])
            )
            gyro_sensor = Sensor(
                'gyroscope', sensor_combined,
                ['gyro_rad[0]', 'gyro_rad[1]', 'gyro_rad[2]'],
                alpha=np.ones(3)*GYRO_LPF_ALPHA
            )
            mag_sensor = Sensor(
                'magnetometer', mag_topic_obj,
                ['x', 'y', 'z'],
                alpha=np.ones(3)*MAG_LPF_ALPHA
            )
            baro_sensor = Sensor(
                'barometer', baro_topic_obj,
                ['pressure'],
                alpha=np.ones(1)*BARO_LPF_ALPHA,
                enable_extrapolation=False
            )

            # Iterate through EKF times
            for idx, t_ekf in enumerate(ekf_time):
                s_idx = np.argmin(np.abs(px4_state_time - t_ekf))
                if abs(px4_state_time[s_idx] - t_ekf) > time_tolerance:
                    continue

                px4_state_vec = px4_states[s_idx]

                # Get sensor data using Sensor.get_data()
                accel_vec_filt = accel_sensor.get_data(t_ekf)
                gyro_vec_filt = gyro_sensor.get_data(t_ekf)
                mag_vec_filt = mag_sensor.get_data(t_ekf)
                baro_val_filt = baro_sensor.get_data(t_ekf)

                # NN input: EKF state + filtered PX4 sensors
                sensor_vec = np.concatenate([
                    accel_vec_filt,
                    gyro_vec_filt,
                    baro_val_filt,
                    mag_vec_filt
                ])
                nn_input = np.concatenate([ekf_states[idx], sensor_vec])

                # NN target: residual = PX4 state - EKF state
                residual = px4_state_vec - ekf_states[idx]

                all_inputs.append(nn_input)
                all_targets.append(residual)
                all_times.append(t_ekf)

        self.inputs = np.array(all_inputs, dtype=np.float32)
        self.targets = np.array(all_targets, dtype=np.float32)
        self.valid_times = np.array(all_times, dtype=np.float32)
        self.seq_len = seq_len

    def __len__(self):
        return self.inputs.shape[0] if self.seq_len == 1 else self.inputs.shape[0] - (self.seq_len - 1)

    def __getitem__(self, idx):
        if self.seq_len == 1:
            return torch.tensor(self.inputs[idx], dtype=torch.float32), torch.tensor(self.targets[idx], dtype=torch.float32)
        else:
            x_seq = torch.tensor(self.inputs[idx:idx+self.seq_len], dtype=torch.float32)
            y_seq = torch.tensor(self.targets[idx:idx+self.seq_len], dtype=torch.float32)
            return x_seq, y_seq

# Neural Network to estimate residual error between EKF and PX4 estimate
class Residual_Estimator(nn.Module): 
    def __init__(self, num_inputs, num_outputs, num_units, num_layers):
        # num_inputs : number of inputs
        # num_outputs: number of outputs
        # num_units  : number of neurons
        # num_layers : Number of hidden layers
        super(Residual_Estimator,self).__init__()
        self.fcI = nn.Linear(num_inputs, num_units) # Input layer
        self.fcs = nn.ModuleList([
            nn.Linear(num_units, num_units) for i in range(num_layers)
        ]) # hidden layers
        self.fcO = nn.Linear(num_units, num_outputs) # Output layers
        
    def forward(self,X):
        # Activation function
        X = torch.relu(self.fcI(X))
        for fcH in self.fcs:
            X = torch.relu(fcH(X))
        X = self.fcO(X)
        return X
    
    def np2torch(self,x):
        x = torch.from_numpy(x).to(device)
        x = x.type(torch.FloatTensor).to(device)
        return x


if __name__=="__main__":
    print("Using device: ", device)
    
    # List of (EKF CSV file, PX4 .ulg file) pairs
    train_ekf_ulog_pairs = [
        (f'./ekf_data/{filename}_orientation_ekf_full_states.csv',
         f'./ekf_data/{filename}_position_ekf_full_states.csv', 
         f'./flight_data/{filename}.ulg')
         for filename in TRAINING_FILES
    ]

    test_ekf_ulog_pairs = [
        (f'./ekf_data/{filename}_orientation_ekf_full_states.csv',
         f'./ekf_data/{filename}_position_ekf_full_states.csv', 
         f'./flight_data/{filename}.ulg')
         for filename in TEST_FILES
    ]

    # Create dataset and data loader
    print("Creating training dataset...")
    train_dataset = EKFResidualDataset(train_ekf_ulog_pairs, time_tolerance=0.03)
    print("Creating test dataset...")
    test_dataset = EKFResidualDataset(test_ekf_ulog_pairs, time_tolerance=0.03)
    train_dataset.targets = smooth_targets(train_dataset.targets, window=10)

    train_inputs = train_dataset.inputs     # shape: (N, num_features)
    train_targets = train_dataset.targets   # shape: (N, num_outputs)

    # Calculate mean and std for inputs
    input_mean = train_inputs.mean(axis=0)
    input_std = train_inputs.std(axis=0) + 1e-8  # avoid division by zero

    # Calculate mean and std for outputs
    target_mean = train_targets.mean(axis=0)
    target_std = train_targets.std(axis=0) + 1e-8

    # Normalize training data
    train_dataset.inputs = (train_dataset.inputs - input_mean) / input_std
    train_dataset.targets = (train_dataset.targets - target_mean) / target_std

    # Normalize test data
    test_dataset.inputs = (test_dataset.inputs - input_mean) / input_std
    test_dataset.targets = (test_dataset.targets - target_mean) / target_std

    # Save training distribution
    np.savez("neural_net/normalization_stats.npz",
         input_mean=input_mean,
         input_std=input_std,
         target_mean=target_mean,
         target_std=target_std)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model init
    num_inputs = train_dataset.inputs.shape[1]   # 20 features: EKF state + PX4 sensors
    num_outputs = train_dataset.targets.shape[1] # 10 features: residual (pos+vel+quat)
    model = Residual_Estimator(
        num_inputs, num_outputs, num_units=NUM_NEURONS, num_layers=NUM_LAYERS
    ).to(device)

    # Optimizer and Loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    def criterion(prediction, batch):
        position_loss = nn.MSELoss()(prediction[:, :3], batch[:, :3])
        velocity_loss = nn.MSELoss()(prediction[:, 3:6], batch[:, 3:6])
        quat_loss = nn.SmoothL1Loss()(prediction[:, 6:], batch[:, 6:])
        return position_loss + velocity_loss + quat_loss

    # Training loop
    epochs = NUM_EPOCHS
    train_losses = []
    test_losses = []

    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        model.train()
        total_train_loss = 0.0

        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Train", leave=False):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * X_batch.size(0)
        
        avg_train_loss = total_train_loss / len(train_dataset)
        train_losses.append(avg_train_loss)

        model.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} - Test", leave=False):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                total_test_loss += loss.item() * X_batch.size(0)
        avg_test_loss = total_test_loss / len(test_dataset)
        test_losses.append(avg_test_loss)

        tqdm.write(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f} - Test Loss: {avg_test_loss:.6f}")
    
    print("Training completed")

    # Save only the model's learned parameters
    torch.save(model.state_dict(), 'neural_net/residual_estimator_model.pth')

    # save the entire model (including architecture)
    torch.save(model, 'neural_net/residual_estimator_model_full.pth')

    print("NN Model saved")

    # Plot loss evolution
    plt.figure(figsize=(8,5))
    plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs+1), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss Evolution')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('neural_net/loss_evolution.png')

    model.eval()
    num_samples_to_plot = 100

    # Prepare inputs and targets from test dataset
    X_plot = torch.tensor(test_dataset.inputs[:num_samples_to_plot], dtype=torch.float32).to(device)
    y_true = test_dataset.targets[:num_samples_to_plot]

    # Predict residuals with the NN
    with torch.no_grad():
        y_pred = model(X_plot).cpu().numpy()

    # Labels for residual dimensions
    residual_labels = [
        'pos_x', 'pos_y', 'pos_z',
        'vel_x', 'vel_y', 'vel_z',
        'quat_w', 'quat_x', 'quat_y', 'quat_z'
    ]

    # Plotting comparison for each residual component
    plt.figure(figsize=(16, 12))
    for i in range(len(residual_labels)):
        plt.subplot(5, 2, i+1)
        plt.plot(y_true[:, i], label='True Residual', alpha=0.7)
        plt.plot(y_pred[:, i], label='NN Predicted Residual', alpha=0.7)
        plt.title(residual_labels[i])
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.suptitle('Comparison of Neural Network Residual Predictions and True Residuals', y=1.02)
    plt.savefig('neural_net/residual_comparison.png')
    plt.show()

