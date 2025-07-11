#establish the connection between gate infidelity and energy cost
#if global depolarizing, convert the noise parameter to gate infidelity
#if local depolarizing, use directly the noise parameter

#functions for gate infidelity and energetics
#can be flexible to incorporate other technologies
#fit function describing the relationship between gate infidelity and energy cost
#total energy as a function of provided number of gates, noise parameter, and noise model - global or local

import numpy as np
from scipy.constants import hbar, k

#results in this case is a tuple (eps, Ng)
class Convert_energy():
    def __init__(self, results, noise_model='global', hardware="Superconducting", control_parameters=None):
        self.hardware = hardware
        self.noise_model = noise_model
        self.noise_param = results[0]  # noise parameter (depolarizing rate)
        self.Ng = results[1]
        print("Noise parameter: ", self.noise_param, " Number of gates: ", self.Ng)
        self.hardware_energy_consumption = 0
        self.gate_infidelity = 0
        #initialization of the control parameters if not provided
        if hardware=="Superconducting":
            if control_parameters is None:
                self.control_parameters = {
                    'Tqb' : 10**(-3), #qubit temperature in K,
                    'Text' : 300, #environment temperature in K
                    'T1qb' : 50*10**(-9), #single qubit gate time in s
                    'gamma' : 1, #spontaneous emission rate in kHz
                    'omega' : 6*10**9, #qubit frequency in Hz
                    'eta' : 0.3, #efficiency of the cryostat wrt carnot efficiency
                    'Adb' : 50 #attenuation in dB
                    }
        else:
            self.control_parameters = control_parameters
        self.algo_resources_to_energy_cost()

    def algo_resources_to_energy_cost(self):
        """
        Convert algorithmic resources to energy cost based on gate infidelity and noise model.
        
        Parameters:
        Ng (int): Number of gates.
        noise_param (float): Noise parameter (depolarizing rate).
        noise_model (str): Type of noise model ('global' or 'local').
        
        Returns:
        float: Estimated energy cost in Joules based on superconducting hardware.
        """
        if self.noise_model == 'global':
            # Convert depolarizing rate to gate infidelity
            self.gate_infidelity = (1- (np.log10(1 - self.noise_param) / self.Ng))/2
        elif self.noise_model == 'local':
            self.gate_infidelity = self.noise_param
        else:
            raise ValueError("Invalid noise model. Choose 'global' or 'local'.")
        print(self.gate_infidelity, " is the gate infidelity")
        if self.hardware == "Superconducting":
            energy_cost = self.superconducting_hardware_resource()
        else:
            raise ValueError("provide detailed hardware description")
        return energy_cost

    def superconducting_hardware_resource(self):
        """
        Calculate the hardware energy consumption for superconducting qubits.
        
        Parameters:
        Ng (int): Number of gates.
        
        Returns:
        float: Total energy cost in Joules.
        """
        Adb = self.control_parameters['Adb']
        A = 10**(Adb/10)  # Convert dB to scalar
        T_qb = self.control_parameters['Tqb']
        T_ext = self.control_parameters['Text']
        omg = self.control_parameters['omega']
        n_qb = (A-1)/(A*(np.exp((hbar*omg)/(k*T_qb))-1))
        n_ext = 1/((np.exp((hbar*omg)/(k*T_ext))-1)*A )
        # Energy per gate in J
        E_1qb = hbar * omg * (np.pi**2) * (1 + n_qb + n_ext) / (4 * self.gate_infidelity)
        #dressed energy consumption for all gates
        E_dressed = (T_ext - T_qb) * E_1qb * self.Ng / (self.control_parameters['eta'] * T_qb)
        self.hardware_energy_consumption = E_dressed
        return self.hardware_energy_consumption
    

    params = (10**-4, 100)
ress = Convert_energy(params)
ress.hardware_energy_consumption