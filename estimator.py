# ---- Imports ----
import csv
import numpy as np
from signal_filters import LowPassFilter
from typing import Callable, Any
from abc import ABC, abstractmethod
from parameters import (
    GRAVITATION_ACCELERATION_ms2, PRESSURE_SEA_LEVEL_Pa, ROOM_TEMPERATURE_K,
    UNIVERSAL_GAS_CONSTANT_JmolK, AIR_MOLAR_MASS_kgmol,
    ORIENTATION_LPF_ALPHA, POSITION_LPF_ALPHA,
)

class Extended_Kalman_Filter():
    def __init__(
            self, state_dict: dict, input_dict: dict, observation_dict: dict,
            process_noise_covariance_Q: np.ndarray, measurement_noise_covariance_R: np.ndarray,
            state_equilibrium=None, input_equilibrium=None, observation_equilibrium=None,
            lpf_alpha=None,
        ):
        self.state_history = {k: v.copy() for k, v in state_dict.items()}
        self.state_history["time"] = []
        self.current_state = np.array([v[0] for v in state_dict.values()])

        self.input_history = {k: v.copy() for k, v in input_dict.items()}
        self.current_input = np.array([v[0] for v in input_dict.values()])

        self.observation_history = {k: v.copy() for k, v in observation_dict.items()}
        self.current_observation = np.array([v[0] for v in observation_dict.values()])

        self.process_noise_covariance_Q = process_noise_covariance_Q
        self.measurement_noise_covariance_R = measurement_noise_covariance_R

        self.num_states = len(self.current_state)
        self.estimation_covariance_P = np.eye(N=self.num_states)

        if state_equilibrium is not None:
            self.state_equilibrium = state_equilibrium
        else:
            self.state_equilibrium = np.zeros_like(self.current_state)

        if input_equilibrium is not None:
            self.input_equilibrium = input_equilibrium
        else:
            self.input_equilibrium = np.zeros_like(self.current_input)

        if observation_equilibrium is not None:
            self.observation_equilibrium = observation_equilibrium
        else:
            self.observation_equilibrium = np.zeros_like(self.current_observation)
        
        self.filter = LowPassFilter(lpf_alpha) if lpf_alpha is not None else None
        
        # Initialize tracking dictionaries for diagonal elements
        self.estimation_covariance_diag_history = {k: [] for k in state_dict.keys()}
        self.innovation_covariance_diag_history = {k: [] for k in observation_dict.keys()}

    @abstractmethod
    def get_state_transition_matrix_A(self) -> np.ndarray:
        pass

    @abstractmethod
    def predict(self) -> None:
        pass

    @abstractmethod
    def observation_model(self, state) -> np.ndarray:
        pass

    def get_observation_matrix_H(self) -> np.ndarray:
        H = self.numerical_jacobian(self.observation_model, self.current_state)
        return H

    def predict_estimation_covariance(self) -> None:
        state_transition_matrix = self.get_state_transition_matrix_A()
        self.estimation_covariance_P = (
            state_transition_matrix @ self.estimation_covariance_P @ state_transition_matrix.T + self.process_noise_covariance_Q
        )
    
    def update_estimation_covariance(self) -> None:
        I = np.eye(self.num_states)
        K = self.get_kalman_gain_K()
        H = self.get_observation_matrix_H()
        R = self.measurement_noise_covariance_R

        self.estimation_covariance_P = (
            (I - K @ H) @ self.estimation_covariance_P @ (I - K @ H).T +
            K @ R @ K.T
        )

    def get_kalman_gain_K(self) -> np.ndarray:
        current_observation_matrix_H = self.get_observation_matrix_H()
        innovation_covariance = (
            current_observation_matrix_H @ self.estimation_covariance_P @ current_observation_matrix_H.T + 
            self.measurement_noise_covariance_R
        )
        kalman_gain_K = (
            self.estimation_covariance_P @ current_observation_matrix_H.T @ np.linalg.inv(innovation_covariance)
        )
        return kalman_gain_K        

    def set_observation_and_input(self, new_observation, new_input) -> None:
        self.current_observation = new_observation
        self.current_input = new_input

    def update(self) -> None:
        self.check_covariance_matrix_P()
        current_observation_matrix_H = self.get_observation_matrix_H()
        kalman_gain_K = self.get_kalman_gain_K()
        self.current_state = (
            self.current_state + 
            kalman_gain_K @ (
                self.current_observation - self.observation_model(self.current_state)
            )
        )
        self.run_low_pass_filter()
        self.update_estimation_covariance()

    def update_ekf_history(self, time=None) -> None:
        for idx, key in enumerate(self.state_history):
            if key != "time":
                self.state_history[key].append(self.current_state[idx])
        for idx, key in enumerate(self.input_history):
            self.input_history[key].append(self.current_input[idx])
        for idx, key in enumerate(self.observation_history):
            self.observation_history[key].append(self.current_observation[idx])
        
        if time is not None:
            self.state_history["time"].append(time)
        else:
            self.state_history["time"].append(None)
        
        # Log diagonal of estimation covariance P
        state_keys = [k for k in self.state_history.keys() if k != "time"]
        for idx, key in enumerate(state_keys):
            self.estimation_covariance_diag_history[key].append(self.estimation_covariance_P[idx, idx])
        
        # Log diagonal of innovation covariance S
        current_observation_matrix_H = self.get_observation_matrix_H()
        innovation_covariance = (
            current_observation_matrix_H @ self.estimation_covariance_P @ current_observation_matrix_H.T + 
            self.measurement_noise_covariance_R
        )
        obs_keys = list(self.observation_history.keys())
        for idx, key in enumerate(obs_keys):
            self.innovation_covariance_diag_history[key].append(innovation_covariance[idx, idx])

    def export(self, path_prefix="ekf_output") -> None:
        def dict_to_csv(dictionary, filename):
            keys = list(dictionary.keys())
            rows = zip(*(dictionary[key] for key in keys))
            with open(filename, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(keys)
                for row in rows:
                    writer.writerow(row)
        dict_to_csv(self.state_history, f"{path_prefix}_states.csv")
        dict_to_csv(self.input_history, f"{path_prefix}_inputs.csv")
        dict_to_csv(self.observation_history, f"{path_prefix}_observations.csv")
        dict_to_csv(self.estimation_covariance_diag_history, f"{path_prefix}_estimation_covariance_diag.csv")
        dict_to_csv(self.innovation_covariance_diag_history, f"{path_prefix}_innovation_covariance_diag.csv")

    def check_covariance_matrix_P(self) -> None:
        """Check if the covariance matrix P is symmetric and positive semi-definite."""
        P = self.estimation_covariance_P

        # Symmetry check
        if not np.allclose(P, P.T, atol=1e-8):
            print(f"Warning: Covariance matrix P is not symmetric at timestep! Max asymmetry: {np.max(np.abs(P - P.T))}")

        # Positive semi-definiteness check
        eigvals = np.linalg.eigvalsh(P)
        if np.any(eigvals < -1e-10):  # Allow tiny negative values from round-off
            print(f"Warning: Covariance matrix P lost PSD, negative eigenvalues: {eigvals[eigvals < 0]}")
    
    def update_state(self, new_state) -> None:
        if len(new_state) == self.num_states:
            for idx, state in enumerate(new_state):
                self.current_state[idx] = state
    
    def run_low_pass_filter(self) -> None:
        if self.filter is not None:
            self.current_state = self.filter.update(self.current_state)

    def numerical_jacobian(self, f, x, eps=1e-7, *args):
        x = np.asarray(x, dtype=float)
        f0 = f(x, *args)
        m = f0.size
        n = x.size
        J = np.zeros((m, n))
        for j in range(n):
            x_perturb = x.copy()
            x_perturb[j] += eps
            f1 = f(x_perturb, *args)
            J[:, j] = (f1 - f0) / eps
        return J

class Orientation_EKF(Extended_Kalman_Filter):
    def __init__(
            self, gyroscope_reading, accelerometer_reading, magnetometer_reading, 
            prediction_timestep, initial_quaternion
            ):
        self.GRAVITATION_ACCELERATION_ms2 = GRAVITATION_ACCELERATION_ms2 # [m/s^2]
        self.LOCAL_MAGNETIC_NORTH = self.rot_mat_from_quat(initial_quaternion) @ magnetometer_reading
        print(f"Local Magnetic North Vector: {self.LOCAL_MAGNETIC_NORTH}")
        # initial reading in NED is set as North, [gauss]
        self.PREDICTION_TIMESTEP_s = prediction_timestep # [seconds]

        state_equilibrium = np.array([
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            # qw, qx, qy, qz, bg_x, bg_y, bg_z, bm_x, bm_y, bm_z
        ])

        input_equilibrium = np.array([0.0, 0.0, 0.0])

        observation_equilibrium = np.array([
            0.0, 0.0, -self.GRAVITATION_ACCELERATION_ms2,
            self.LOCAL_MAGNETIC_NORTH[0],
            self.LOCAL_MAGNETIC_NORTH[1],
            self.LOCAL_MAGNETIC_NORTH[2]
        ])

        state_dict = {
            "quaternion (w)": [initial_quaternion[0]],
            "quaternion (x)": [initial_quaternion[1]],
            "quaternion (y)": [initial_quaternion[2]],
            "quaternion (z)": [initial_quaternion[3]],
            "gyroscope bias (x)": [0.0],
            "gyroscope bias (y)": [0.0],
            "gyroscope bias (z)": [0.0],
            "magnetometer bias (x)": [0.0],
            "magnetometer bias (y)": [0.0],
            "magnetometer bias (z)": [0.0],
        }
        input_dict = {
            "gyroscope (x)": [gyroscope_reading[0]],
            "gyroscope (y)": [gyroscope_reading[1]],
            "gyroscope (z)": [gyroscope_reading[2]]
        }
        observation_dict = {
            "accelerometer (x)": [accelerometer_reading[0]],
            "accelerometer (y)": [accelerometer_reading[1]],
            "accelerometer (z)": [accelerometer_reading[2]],
            "magnetometer (x)": [magnetometer_reading[0]],
            "magnetometer (y)": [magnetometer_reading[1]],
            "magnetometer (z)": [magnetometer_reading[2]]   
        }
        process_noise_covariance_Q = np.diag([
            1.0e-8, # qw
            1.0e-8, # qx
            1.0e-8, # qy
            1.0e-8, # qz
            1.0e-8, # gyro bias (x)
            1.0e-8, # gyro bias (y)
            1.0e-8, # gyro bias (z)
            1.0e-8, # magnetometer bias (x)
            1.0e-8, # magnetometer bias (y)
            1.0e-8, # magnetometer bias (z)
        ])
        measurement_noise_covariance_R = np.diag([
            5.0e5, # accelerometer (x)
            5.0e5, # accelerometer (y)
            5.0e5, # accelerometer (z)
            5.0e5, # magnetometer (x)
            5.0e5, # magnetometer (y)
            5.0e5, # magnetometer (z)
        ])
        super().__init__(
            state_dict, input_dict, observation_dict, 
            process_noise_covariance_Q, measurement_noise_covariance_R,
            state_equilibrium, input_equilibrium, observation_equilibrium,
            lpf_alpha=None
        )

    def rot_mat_from_quat(self, q) -> np.ndarray:
        rotation_matrix = np.array([
            [1 - 2*q[2]**2 - 2*q[3]**2, 2*q[1]*q[2] - 2*q[3]*q[0], 2*q[1]*q[3] + 2*q[2]*q[0]],
            [2*q[1]*q[2] + 2*q[3]*q[0], 1 - 2*q[1]**2 - 2*q[3]**2, 2*q[2]*q[3] - 2*q[1]*q[0]],
            [2*q[1]*q[3] - 2*q[2]*q[0], 2*q[2]*q[3] + 2*q[1]*q[0], 1 - 2*q[1]**2 - 2*q[2]**2]
        ])
        return rotation_matrix
    
    def observation_model(self, state):
        R = self.rot_mat_from_quat(state[:4])
        predicted_observation = np.vstack([
            R.T @ np.array([0, 0, -self.GRAVITATION_ACCELERATION_ms2]),
              # accelerometer
            R.T @ self.LOCAL_MAGNETIC_NORTH  # magnetometer
        ])
        return predicted_observation.flatten()

    def get_state_transition_matrix_A(self) -> np.ndarray:
        dt = self.PREDICTION_TIMESTEP_s
        q_w, q_x, q_y, q_z, bg_x, bg_y, bg_z, bm_x, bm_y, bm_z = self.current_state
        w_x, w_y, w_z = self.current_input

        state_transition_matrix = np.array([
            [1, dt*(0.5*bg_x - 0.5*w_x), dt*(0.5*bg_y - 0.5*w_y), dt*(0.5*bg_z - 0.5*w_z), 0.5*dt*q_x, 0.5*dt*q_y, 0.5*dt*q_z, 0, 0, 0], 
            [dt*(-0.5*bg_x + 0.5*w_x), 1, dt*(-0.5*bg_z + 0.5*w_z), dt*(0.5*bg_y - 0.5*w_y), -0.5*dt*q_w, 0.5*dt*q_z, -0.5*dt*q_y, 0, 0, 0], 
            [dt*(-0.5*bg_y + 0.5*w_y), dt*(0.5*bg_z - 0.5*w_z), 1, dt*(-0.5*bg_x + 0.5*w_x), -0.5*dt*q_z, -0.5*dt*q_w, 0.5*dt*q_x, 0, 0, 0], 
            [dt*(-0.5*bg_z + 0.5*w_z), dt*(-0.5*bg_y + 0.5*w_y), dt*(0.5*bg_x - 0.5*w_x), 1, 0.5*dt*q_y, -0.5*dt*q_x, -0.5*dt*q_w, 0, 0, 0], 
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
            ], dtype=float)
        return state_transition_matrix

    def predict(self) -> None:
        q_w, q_x, q_y, q_z, bg_x, bg_y, bg_z, bm_x, bm_y, bm_z = self.current_state
        w_x, w_y, w_z = self.current_input
        dt = self.PREDICTION_TIMESTEP_s

        q_w_next = dt*(q_x*(0.5*bg_x - 0.5*w_x) + q_y*(0.5*bg_y - 0.5*w_y) + q_z*(0.5*bg_z - 0.5*w_z)) + q_w
        q_x_next = dt*(q_w*(-0.5*bg_x + 0.5*w_x) + q_y*(-0.5*bg_z + 0.5*w_z) + q_z*(0.5*bg_y - 0.5*w_y)) + q_x
        q_y_next = dt*(q_w*(-0.5*bg_y + 0.5*w_y) + q_x*(0.5*bg_z - 0.5*w_z) + q_z*(-0.5*bg_x + 0.5*w_x)) + q_y 
        q_z_next = dt*(q_w*(-0.5*bg_z + 0.5*w_z) + q_x*(-0.5*bg_y + 0.5*w_y) + q_y*(0.5*bg_x - 0.5*w_x)) + q_z
        
        self.current_state = np.array([q_w_next, q_x_next, q_y_next, q_z_next, bg_x, bg_y, bg_z, bm_x, bm_y, bm_z])
        self.normalize_quaternion()

        self.predict_estimation_covariance()

    def normalize_quaternion(self) -> None:
        qw, qx, qy, qz = self.current_state[:4]
        length_of_quaternion = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
        qw_new = qw / length_of_quaternion
        qx_new = qx / length_of_quaternion
        qy_new = qy / length_of_quaternion
        qz_new = qz / length_of_quaternion
        self.current_state[:4] = [qw_new, qx_new, qy_new, qz_new]

    def update(self) -> None:
        super().update()
        self.normalize_quaternion()

class Position_Velocity_EKF(Extended_Kalman_Filter):
    def __init__(self, accelerometer_reading, barometer_reading, prediction_timestep, initial_quaternion):
        self.orientation_state = np.array([
            initial_quaternion[0] if initial_quaternion is not None else 1, # qw
            initial_quaternion[1] if initial_quaternion is not None else 0, # qx
            initial_quaternion[2] if initial_quaternion is not None else 0, # qy
            initial_quaternion[3] if initial_quaternion is not None else 0, # qz
        ], dtype=float)

        self.GRAVITATION_ACCELERATION_ms2 = GRAVITATION_ACCELERATION_ms2
        # Acceleration due to gravity [m/s^2]
        self.PREDICTION_TIMESTEP = prediction_timestep
 
        # Atmospheric Parameters (for baro)
        self.PRESSURE_SEA_LEVEL_Pa = PRESSURE_SEA_LEVEL_Pa 
        # Sea level standard atmospheric pressure [Pa]
        self.ROOM_TEMPERATURE_K = ROOM_TEMPERATURE_K 
        # Standard room temperature [K]
        self.UNIVERSAL_GAS_CONSTANT_JmolK = UNIVERSAL_GAS_CONSTANT_JmolK
        # Universal gas constant [J/(mol*K)]
        self.AIR_MOLAR_MASS_kgmol = AIR_MOLAR_MASS_kgmol 
        # Molar mass of dry air [kg/mol]

        state_equilibrium = np.array([
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            # px, py, pz, vx, vy, vz
        ])
        
        input_equilibrium = np.array([
            0.0, 0.0, -self.GRAVITATION_ACCELERATION_ms2, 
        ])

        observation_equilibrium = barometer_reading

        state_dict = {
            "position (x) [m]": [0.0],
            "position (y) [m]": [0.0],
            "position (z) [m]": [0.0], # altitude above msl
            "velocity (x) [m/s]": [0.0],
            "velocity (y) [m/s]": [0.0],
            "velocity (z) [m/s]": [0.0],
        }
        input_dict = {
            "accelerometer (x)": [accelerometer_reading[0]],
            "accelerometer (y)": [accelerometer_reading[1]],
            "accelerometer (z)": [accelerometer_reading[2]],
        }
        observation_dict = {
            "barometer pressure [Pa]": [barometer_reading[0]],
        }
        process_noise_covariance_Q = np.diag([
            1.0e5, # px
            1.0e5, # py
            1.0e2, # pz
            1.0e-3, # vx
            1.0e-3, # vy
            1.0e-3, # vz
        ])
        measurement_noise_covariance_R = np.diag([
            1.0e-5, # baro pressure
        ])
        super().__init__(
            state_dict, input_dict, observation_dict, 
            process_noise_covariance_Q, measurement_noise_covariance_R,
            state_equilibrium, input_equilibrium, observation_equilibrium,
            lpf_alpha=POSITION_LPF_ALPHA,
        )

    def update_orientation(self, orientation: np.ndarray) -> None:
        self.orientation_state = orientation

    def get_state_transition_matrix_A(self) -> np.ndarray: #TODO
        dt = self.PREDICTION_TIMESTEP
        state_transition_matrix = np.array([
            [1, 0, 0, dt, 0, 0], 
            [0, 1, 0, 0, dt, 0], 
            [0, 0, 1, 0, 0, dt], 
            [0, 0, 0, 1, 0, 0], 
            [0, 0, 0, 0, 1, 0], 
            [0, 0, 0, 0, 0, 1], 
            ], dtype=float)
        return state_transition_matrix

    def observation_model(self, state) -> np.ndarray:
        altitude_m = state[2]
        P_0 = self.PRESSURE_SEA_LEVEL_Pa
        R = self.UNIVERSAL_GAS_CONSTANT_JmolK
        M = self.AIR_MOLAR_MASS_kgmol
        g = self.GRAVITATION_ACCELERATION_ms2
        T_r = self.ROOM_TEMPERATURE_K

        predicted_observation = np.array([
            P_0 * np.exp(M * g * altitude_m / (T_r * R))
        ])
        return predicted_observation
    
    def predict(self) -> None:
        p_x, p_y, p_z, v_x, v_y, v_z = self.current_state
        a_x, a_y, a_z = self.current_input
        q_w, q_x, q_y, q_z = self.orientation_state

        dt = self.PREDICTION_TIMESTEP
        g = self.GRAVITATION_ACCELERATION_ms2

        p_x_next = (
            dt**2*(0.5*a_x*(-2*q_y**2 - 2*q_z**2 + 1) + 
                   0.5*a_y*(-2*q_w*q_z + 2*q_x*q_y) + 0.5*a_z*(2*q_w*q_y + 2*q_x*q_z)) + 
                   dt*v_x + p_x
        )
        p_y_next = (
            dt**2*(0.5*(a_x)*(2*q_w*q_z + 2*q_x*q_y) + 
                   0.5*(a_y)*(-2*q_x**2 - 2*q_z**2 + 1) + 0.5*(a_z)*(-2*q_w*q_x + 2*q_y*q_z)) + dt*v_y + p_y
        )
        p_z_next = (
            dt**2*(0.5*g + 0.5*(a_x)*(-2*q_w*q_y + 2*q_x*q_z) + 
                   0.5*(a_y)*(2*q_w*q_x + 2*q_y*q_z) + 0.5*(a_z)*(-2*q_x**2 - 2*q_y**2 + 1)) + dt*v_z + p_z
        ) 
        v_x_next = (
            dt*((a_x)*(-2*q_y**2 - 2*q_z**2 + 1) + 
                (a_y)*(-2*q_w*q_z + 2*q_x*q_y) + (a_z)*(2*q_w*q_y + 2*q_x*q_z)) + v_x
        )
        v_y_next = (
            dt*((a_x)*(2*q_w*q_z + 2*q_x*q_y) + 
                (a_y)*(-2*q_x**2 - 2*q_z**2 + 1) + (a_z)*(-2*q_w*q_x + 2*q_y*q_z)) + v_y
        )
        v_z_next = (
            dt*(g + (a_x)*(-2*q_w*q_y + 2*q_x*q_z) + 
                (a_y)*(2*q_w*q_x + 2*q_y*q_z) + (a_z)*(-2*q_x**2 - 2*q_y**2 + 1)) + v_z
        )

        self.current_state = np.array([p_x_next, p_y_next, p_z_next, v_x_next, v_y_next, v_z_next])
        self.predict_estimation_covariance()

