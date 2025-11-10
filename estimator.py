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

    @abstractmethod
    def get_state_transition_matrix_A(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_observation_matrix_H(self) -> np.ndarray:
        pass

    @abstractmethod
    def predict(self) -> None:
        pass

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
        kalman_gain_K = (
            self.estimation_covariance_P @ current_observation_matrix_H.T @ np.linalg.inv(
                current_observation_matrix_H @ self.estimation_covariance_P @ current_observation_matrix_H.T + self.measurement_noise_covariance_R
            )
        )
        return kalman_gain_K        

    def set_observation(self, new_observation, new_input) -> None:
        self.current_observation = new_observation
        self.current_input = new_input

    def update(self) -> None:
        self.check_covariance_matrix_P()
        current_observation_matrix_H = self.get_observation_matrix_H()
        kalman_gain_K = self.get_kalman_gain_K()
        self.current_state = (
            self.current_state + 
            kalman_gain_K @ ((self.current_observation - self.observation_equilibrium) - current_observation_matrix_H @ self.current_state)
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
            # For missing time info, append None or a placeholder
            self.state_history["time"].append(None)

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

class Orientation_EKF(Extended_Kalman_Filter):
    def __init__(
            self, gyroscope_reading, accelerometer_reading, magnetometer_reading, 
            prediction_timestep, initial_quaternion=None
            ):
        self.GRAVITATION_ACCELERATION_ms2 = GRAVITATION_ACCELERATION_ms2 # [m/s^2]
        self.LOCAL_MAGNETIC_FIELD_VECTOR = magnetometer_reading 
        # initial reading is set as North, [gauss]
        self.PREDICTION_TIMESTEP_s = prediction_timestep # [seconds]

        state_equilibrium = np.array([
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            # qw, qx, qy, qz, bg_x, bg_y, bg_z, bm_x, bm_y, bm_z
        ])

        input_equilibrium = np.array([0.0, 0.0, 0.0])

        observation_equilibrium = np.array([
            0.0, 0.0, -self.GRAVITATION_ACCELERATION_ms2,
            magnetometer_reading[0], magnetometer_reading[1], magnetometer_reading[2]
        ])

        state_dict = {
            "quaternion (w)": [initial_quaternion[0] if initial_quaternion is not None else 1.0],
            "quaternion (x)": [initial_quaternion[1] if initial_quaternion is not None else 0.0],
            "quaternion (y)": [initial_quaternion[2] if initial_quaternion is not None else 0.0],
            "quaternion (z)": [initial_quaternion[3] if initial_quaternion is not None else 0.0],
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
            1.0e-2, # qw
            1.0e-2, # qx
            1.0e-2, # qy
            1.0e-2, # qz
            1.0e1, # gyro bias (x)
            1.0e1, # gyro bias (y)
            1.0e1, # gyro bias (z)
            1.0e1, # magnetometer bias (x)
            1.0e1, # magnetometer bias (y)
            1.0e1, # magnetometer bias (z)
        ])
        measurement_noise_covariance_R = np.diag([
            1.0e-1, # accelerometer (x)
            1.0e-1, # accelerometer (y)
            1.0e-1, # accelerometer (z)
            1.0e-1, # magnetometer (x)
            1.0e-1, # magnetometer (y)
            1.0e-1, # magnetometer (z)
        ])
        super().__init__(
            state_dict, input_dict, observation_dict, 
            process_noise_covariance_Q, measurement_noise_covariance_R,
            state_equilibrium, input_equilibrium, observation_equilibrium,
            lpf_alpha=None
        )

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

    def get_observation_matrix_H(self) -> np.ndarray:
        g = self.GRAVITATION_ACCELERATION_ms2
        m_x, m_y, m_z = self.LOCAL_MAGNETIC_FIELD_VECTOR
        q_w, q_x, q_y, q_z = self.current_state[:4]
        
        observation_matrix = np.array([
            [2*g*q_y, -2*g*q_z, 2*g*q_w, -2*g*q_x, 0, 0, 0, 0, 0, 0], 
            [-2*g*q_x, -2*g*q_w, -2*g*q_z, -2*g*q_y, 0, 0, 0, 0, 0, 0], 
            [0, 4*g*q_x, 4*g*q_y, 0, 0, 0, 0, 0, 0, 0], 
            [
                2*m_y*q_z - 2*m_z*q_y, 
                2*m_y*q_y + 2*m_z*q_z, 
                -4*m_x*q_y + 2*m_y*q_x - 2*m_z*q_w, 
                -4*m_x*q_z + 2*m_y*q_w + 2*m_z*q_x,
                0, 0, 0, 1, 0, 0
                ], 
            [
                -2*m_x*q_z + 2*m_z*q_x, 
                2*m_x*q_y - 4*m_y*q_x + 2*m_z*q_w, 
                2*m_x*q_x + 2*m_z*q_z, 
                -2*m_x*q_w - 4*m_y*q_z + 2*m_z*q_y, 
                0, 0, 0, 0, 1, 0
                ], 
            [
                2*m_x*q_y - 2*m_y*q_x, 
                2*m_x*q_z - 2*m_y*q_w - 4*m_z*q_x, 
                2*m_x*q_w + 2*m_y*q_z - 4*m_z*q_y, 
                2*m_x*q_x + 2*m_y*q_y, 
                0, 0, 0, 0, 0, 1
                ]
            ], dtype=float)
        return observation_matrix

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
            1.0e10, # px
            1.0e10, # py
            1.0e2, # pz
            1.0e-1, # vx
            1.0e-1, # vy
            1.0e6, # vz
        ])
        measurement_noise_covariance_R = np.diag([
            1.0e-15, # baro pressure
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

    def get_observation_matrix_H(self) -> np.ndarray:
        P_0 = self.PRESSURE_SEA_LEVEL_Pa
        R = self.UNIVERSAL_GAS_CONSTANT_JmolK
        M = self.AIR_MOLAR_MASS_kgmol
        g = self.GRAVITATION_ACCELERATION_ms2
        T_r = self.ROOM_TEMPERATURE_K

        p_z = self.current_state[2] # Current altitude [m]

        observation_matrix = np.array([[
            0, 0, 
            M * P_0 * g * np.exp(M * g * p_z / (R * T_r)) / (R * T_r), 
            0, 0, 0
        ]])
        return observation_matrix
    
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

