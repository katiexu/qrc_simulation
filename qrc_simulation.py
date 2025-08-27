#!/usr/bin/env python3
"""
Quantum Reservoir Computing (QRC) Simulation
Refactored from Tutorial.ipynb for easier debugging
"""

import matplotlib.pyplot as plt
import torch as pt
import numpy as np
import pandas as pd
import scipy
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import os
import json
from sklearn.metrics import mean_squared_error

# Import utilities
from utilities import *


class QRCSimulator:
    """Quantum Reservoir Computing Simulator"""
    
    def __init__(self, total_spins=6, input_sys_size=1, T_evolution=10):
        """
        Initialize QRC simulator
        
        Args:
            total_spins: Total number of quantum spins
            input_sys_size: Size of input system 
            T_evolution: Time evolution parameter
        """
        self.total_spins = total_spins
        self.input_sys_size = input_sys_size
        self.T_evolution = T_evolution
        
        # Initialize Hilbert spaces
        self._setup_hilbert_spaces()
        
        # Initialize quantum state
        self._setup_initial_state()
        
        # Setup basis change matrices
        self._setup_basis_changes()
        
        # Setup data structure
        self._setup_data_structure()
        
    def _setup_hilbert_spaces(self):
        """Setup Hilbert spaces for the quantum system"""
        self.spin_space_B = HilbertSpace(self.total_spins - self.input_sys_size, 'spin')
        self.spin_space_A = HilbertSpace(self.input_sys_size, 'spin')
        
        hilbert_spaces_dict = {
            'spin_A': self.spin_space_A,
            'spin_B': self.spin_space_B
        }
        self.composite_space = CompositeHilbertSpace(hilbert_spaces_dict)
        
    def _setup_initial_state(self):
        """Setup initial quantum state |000000>"""
        rho_0_spins = np.zeros((2**self.total_spins, 2**self.total_spins), dtype=np.complex128)
        rho_0_spins[0][0] = 1.0 + 1j*0.0
        self.rho_0 = DensityMatrix(rho_0_spins)
        
    def _setup_basis_changes(self):
        """Setup basis change matrices for X, Y measurements"""
        # Hadamard matrix for X basis
        H = 1/np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=complex)
        H_total = H
        for i in range(self.total_spins - 1):
            H_total = np.kron(H_total, H)
        self.X_basis_change = H_total
        self.X_basis_change_dag = H_total.conj().T
        
        # Phase gate for Y basis
        S = np.array([[1, 0], [0, -np.exp(1j*np.pi/2)]], dtype=complex)
        S_total = S
        for i in range(self.total_spins - 1):
            S_total = np.kron(S_total, S)
        self.Y_basis_change = H_total @ S_total
        self.Y_basis_change_dag = self.Y_basis_change.conj().T
        
    def _setup_data_structure(self):
        """Setup data structure for storing results"""
        column_length_dict = {
            'X': self.total_spins,
            'Y': self.total_spins,
            'Z': self.total_spins,
            'XX': self.total_spins * (self.total_spins + 1) // 2,
            'YY': self.total_spins * (self.total_spins + 1) // 2,
            'ZZ': self.total_spins * (self.total_spins + 1) // 2
        }
        
        self.df_columns = []
        for col in column_length_dict.keys():
            for i in range(column_length_dict[col]):
                self.df_columns.append(col + str(i))
    
    @staticmethod
    def get_couplings(n_spins, J):
        """Generate random coupling matrix"""
        J_bare = np.random.uniform(low=-0.5+J, high=0.5*J, size=(n_spins, n_spins))
        # Exclude self-interaction
        for i in range(n_spins):
            J_bare[i, i] = 0
        # Make symmetric
        J_bare = np.triu(J_bare, k=1)
        J_bare = 0.5 * (J_bare + J_bare.T)
        return J_bare
    
    def get_hamiltonian(self, h=10, J=1):
        """
        Generate reservoir Hamiltonian
        
        Args:
            h: Magnetic field strength
            J: Coupling strength
            
        Returns:
            H_reservoir: Reservoir Hamiltonian matrix
        """
        H_reservoir = np.zeros((self.composite_space.dimension, self.composite_space.dimension))
        J_bare = self.get_couplings(self.total_spins, J)
        
        # XX coupling terms
        for i in range(1, self.total_spins + 1):
            for j in range(i + 1, self.total_spins + 1):
                H_reservoir += J * J_bare[i-1, j-1] * self.composite_space.X[i] @ self.composite_space.X[j]
        
        # Z field terms
        for i in range(1, self.total_spins + 1):
            H_reservoir += 0.5 * h * self.composite_space.Z[i]
            
        return H_reservoir
    
    @staticmethod
    def M_matrix(N, g):
        """Generate measurement backaction matrix"""
        Mi = np.array([[1.0, np.exp(-g**2/2)], [np.exp(-g**2/2), 1]], dtype=np.complex128)
        M = Mi * 1
        for j in range(N - 1):
            M = np.kron(M, Mi)
        return M
    
    def run_simulation(self, time_series, g=0.5, h=1, verbose=True):
        """
        Run QRC simulation for given parameters
        
        Args:
            time_series: Input time series data
            g: Measurement strength parameter
            h: Magnetic field strength
            verbose: Whether to print progress
            
        Returns:
            features_df: DataFrame with extracted features
        """
        if verbose:
            print(f"Running QRC simulation with g={g}, h={h}")
            
        # Initialize Hamiltonian and evolution
        H_reservoir = self.get_hamiltonian(h)
        U_reservoir = scipy.linalg.expm(-1j * H_reservoir * self.T_evolution)
        U_reservoir_dag = U_reservoir.conj().T
        
        # Measurement backaction matrix
        M = self.M_matrix(self.total_spins, g)
        
        # Initialize data storage
        data = {
            'X': [], 'Y': [], 'Z': [],
            'XX': [], 'YY': [], 'ZZ': []
        }
        
        K = len(time_series)
        
        # Process time series for each observable
        for obs in ['X', 'Y', 'Z']:
            if verbose:
                print(f"Processing {obs} observable...")
                
            rho = self.rho_0.dm
            
            for k in tqdm(range(K), disable=not verbose):
                if isinstance(rho, DensityMatrix):
                    rho = rho.dm
                
                # Input encoding
                s_k = time_series[k]
                rho_B = self.composite_space.get_reduced_density_matrix(rho, ['spin_B'])
                rho_A = np.array([
                    [1. - s_k, np.sqrt((1 - s_k) * s_k)],
                    [np.sqrt((1 - s_k) * s_k), s_k]
                ])
                rho = np.kron(rho_A, rho_B)
                
                # Time evolution
                rho = U_reservoir @ rho @ U_reservoir_dag
                
                # Measurement for different observables
                if obs == 'Z':
                    rho = self._measure_Z(rho, M, data)
                elif obs == 'Y':
                    rho = self._measure_Y(rho, M, data)
                elif obs == 'X':
                    rho = self._measure_X(rho, M, data)
        
        # Convert to DataFrame
        data_arr = self._dict_to_arr(data)
        features_df = pd.DataFrame(data_arr, columns=self.df_columns)
        
        if verbose:
            print(f"Simulation completed. Features shape: {features_df.shape}")
            
        return features_df
    
    def _measure_Z(self, rho, M, data):
        """Measure Z observables"""
        # Apply measurement
        rho = np.multiply(M, rho)
        rho = DensityMatrix(rho)
        
        # Single-spin Z expectation values
        ev_Z = []
        for i in range(1, self.total_spins + 1):
            ev_Z.append(rho.get_expectation_value(self.composite_space.Z[i]))
        data['Z'].append(ev_Z)
        
        # Two-spin ZZ correlations
        ev_ZZ = []
        for i in range(1, self.total_spins + 1):
            for j in range(i, self.total_spins + 1):
                ev_ZZ.append(rho.get_expectation_value(
                    self.composite_space.Z[i] @ self.composite_space.Z[j]
                ))
        data['ZZ'].append(ev_ZZ)
        
        return rho.dm
    
    def _measure_Y(self, rho, M, data):
        """Measure Y observables"""
        # Rotate to Y basis
        rho = self.Y_basis_change @ rho @ self.Y_basis_change_dag
        # Apply measurement
        rho = np.multiply(M, rho)
        # Rotate back to Z basis
        rho = self.Y_basis_change_dag @ rho @ self.Y_basis_change
        rho = DensityMatrix(rho)
        
        # Single-spin Y expectation values
        ev_Y = []
        for i in range(1, self.total_spins + 1):
            ev_Y.append(rho.get_expectation_value(self.composite_space.Y[i]))
        data['Y'].append(ev_Y)
        
        # Two-spin YY correlations
        ev_YY = []
        for i in range(1, self.total_spins + 1):
            for j in range(i, self.total_spins + 1):
                ev_YY.append(rho.get_expectation_value(
                    self.composite_space.Y[i] @ self.composite_space.Y[j]
                ))
        data['YY'].append(ev_YY)
        
        return rho.dm
    
    def _measure_X(self, rho, M, data):
        """Measure X observables"""
        # Rotate to X basis
        rho = self.X_basis_change @ rho @ self.X_basis_change_dag
        # Apply measurement
        rho = np.multiply(M, rho)
        # Rotate back to Z basis
        rho = self.X_basis_change_dag @ rho @ self.X_basis_change
        rho = DensityMatrix(rho)
        
        # Single-spin X expectation values
        ev_X = []
        for i in range(1, self.total_spins + 1):
            ev_X.append(rho.get_expectation_value(self.composite_space.X[i]))
        data['X'].append(ev_X)
        
        # Two-spin XX correlations
        ev_XX = []
        for i in range(1, self.total_spins + 1):
            for j in range(i, self.total_spins + 1):
                ev_XX.append(rho.get_expectation_value(
                    self.composite_space.X[i] @ self.composite_space.X[j]
                ))
        data['XX'].append(ev_XX)
        
        return rho.dm
    
    @staticmethod
    def _dict_to_arr(data_dict):
        """Convert data dictionary to structured array"""
        num_lists_per_key = len(data_dict[next(iter(data_dict))])
        concatenated_lists = []
        
        for i in range(num_lists_per_key):
            current_row = []
            for key in data_dict:
                current_row.extend(data_dict[key][i])
            concatenated_lists.append(current_row)
            
        return np.array(concatenated_lists)


class QRCAnalyzer:
    """QRC Performance Analysis Tools"""
    
    @staticmethod
    def compute_capacity(features_df, time_series, eta, f_p=True):
        """
        Compute memory/prediction capacity for given eta
        
        Args:
            features_df: Features DataFrame
            time_series: Input time series
            eta: Time delay parameter
            f_p: True for prediction, False for memory
            
        Returns:
            capacity_train, capacity_test: Training and test capacities
        """
        tot_width = features_df.shape[1]
        N_raw_data = len(features_df)
        N_skip = 20
        data_size = N_raw_data - N_skip
        
        # Prepare data tensors
        X_data = pt.zeros(data_size, tot_width, dtype=float)
        Y_ini_data = pt.zeros(data_size, 1, dtype=float)
        
        for k in range(data_size):
            X_data[k] = pt.tensor(features_df.loc[k, :].values, dtype=float)
            Y_ini_data[k] = time_series[k]
        
        eta = abs(eta)
        if f_p:
            eta = -eta
        
        Y_data = pt.roll(Y_ini_data, -eta)
        
        # Train-test split
        fraction_train = 0.7
        N_train = int(fraction_train * data_size)
        
        X_train = X_data[:N_train, :].float()
        y_train = Y_data[:N_train].float()
        X_test = X_data[-data_size + N_train:, :].float()
        y_test = Y_data[-data_size + N_train:].float()
        
        # Linear regression
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        y_train = y_train.detach().numpy()
        y_test = y_test.detach().numpy()
        
        # Compute capacities
        capacity_train = np.cov(y_train.T, y_train_pred.T, ddof=1)[0, 1]**2 / (
            np.var(y_train, ddof=1) * np.var(y_train_pred, ddof=1)
        )
        
        capacity_test = np.cov(y_test.T, y_test_pred.T, ddof=1)[0, 1]**2 / (
            np.var(y_test, ddof=1) * np.var(y_test_pred, ddof=1)
        )
        
        return capacity_train, capacity_test
    
    @staticmethod
    def compute_sum_capacity(features_df, time_series, eta_max, f_p=True):
        """
        Compute sum capacity up to eta_max
        
        Args:
            features_df: Features DataFrame
            time_series: Input time series
            eta_max: Maximum eta value
            f_p: True for prediction, False for memory
            
        Returns:
            sum_capacity_test: Sum of test capacities
        """
        capacities_test = []
        
        for eta in range(1, eta_max + 1):
            _, capacity_test = QRCAnalyzer.compute_capacity(
                features_df, time_series, eta, f_p
            )
            capacities_test.append(capacity_test)
        
        return np.sum(capacities_test)

    @staticmethod
    def rnn(features_df, time_series,eta,seq_length):

        tot_width = features_df.shape[1]
        N_raw_data = len(features_df)
        N_skip = 20
        data_size = N_raw_data - N_skip-seq_length

        # Prepare data tensors
        X_data = pt.zeros(data_size, seq_length*tot_width, dtype=float)
        Y_ini_data = pt.zeros(data_size, 1, dtype=float)

        data=features_df.values
        for k in range(seq_length,data_size):
            X_data[k] = pt.tensor(data[k-seq_length:k, :].flatten(), dtype=float)
            Y_ini_data[k] = time_series[k]

        eta = abs(eta)

        Y_data = pt.roll(Y_ini_data, -eta)

        fraction_train = 0.7
        N_train = int(fraction_train * data_size)

        X_train = X_data[:N_train, :].float()
        y_train = Y_data[:N_train].float()
        X_test = X_data[-data_size + N_train:, :].float()
        y_test = Y_data[-data_size + N_train:].float()

        # Linear regression
        model = LinearRegression()
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        y_train = y_train.detach().numpy()
        y_test = y_test.detach().numpy()

        # Compute capacities
        capacity_train = np.cov(y_train.T, y_train_pred.T, ddof=1)[0, 1] ** 2 / (
                np.var(y_train, ddof=1) * np.var(y_train_pred, ddof=1)
        )

        capacity_test = np.cov(y_test.T, y_test_pred.T, ddof=1)[0, 1] ** 2 / (
                np.var(y_test, ddof=1) * np.var(y_test_pred, ddof=1)
        )
        mse = mean_squared_error(y_test, y_test_pred)

        return capacity_train, capacity_test,mse


def load_time_series(data_type="santa_fe"):
    """
    Load and preprocess time series data
    
    Args:
        data_type: "santa_fe" or "smt"
        
    Returns:
        time_series: Preprocessed time series
    """
    if data_type == "santa_fe":
        # Forward prediction task
        time_series_raw = np.load("./sk_Santa_Fe_2000.npy")
        min_ts = min(time_series_raw)
        max_ts = max(time_series_raw)
        time_series = (time_series_raw + np.abs(min_ts)) / (max_ts - min_ts)
        time_series = time_series.flatten()
    elif data_type == "smt":
        # Memory retrieval task
        time_series = np.load("./time_series_smt.npy")
    elif data_type == "stock":
        # 读取数据
        data = pd.read_csv('./stock_price/traindata_stock.csv')['Open'].values.reshape(-1, 1)
        # data = pd.read_csv('./stock_price/testdata_stock.csv')['Open'].values.reshape(-1, 1)
        # 初始化归一化器
        train_scaler = MinMaxScaler(feature_range=(0, 1))

        # 对训练数据进行拟合和转换
        train_data_scaled = train_scaler.fit_transform(data)

        # 将归一化后的数据转换回一维数组
        time_series = train_data_scaled.flatten().tolist()
    else:
        raise ValueError("data_type must be 'santa_fe' or 'smt'")
    
    return time_series


def load_figure_params():
    """Load figure parameters from JSON file"""
    with open('./figure_params.json') as json_file:
        return json.load(json_file)


def main():
    """Main function for demonstration"""
    print("QRC Simulation Demo")
    print("==================")
    
    # Load time series
    time_series = load_time_series("stock")
    print(f"Loaded time series with {len(time_series)} points")
    
    # Initialize simulator
    simulator = QRCSimulator()
    
    # Run simulation with default parameters
    g, h = 0.5, 1
    features_df = simulator.run_simulation(time_series, g=g, h=h)

    # 创建序列
    eta=7
    seq_length=6
    capacity_train, capacity_test,mse = QRCAnalyzer().rnn(features_df,time_series, eta,seq_length)

    print(f"eta = {eta} seq_length = {seq_length}  Capacity : {np.round(capacity_test, 4)} \t mse : {mse:.4f}")



if __name__ == "__main__":
    main()
