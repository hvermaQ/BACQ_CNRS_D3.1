#classes for energetic analysis
#takes a list of jobs, results and exact values
#can be intiialized with a keyword for hardware e.g. :superconducting
#the control parameters are available as default, can be changed during intialization
#if only one input, returns only the values
#1. algorithmic energy consumption [{'Ng': int, 't': int, 'Ns' : int, 'Nm':int}]
#2. Hardware energy consumption
#3. Energetic efficiency

import numpy as np
from scipy.constants import hbar

#gateset for counting gates
one_qb_gateset = ['H', 'X', 'Y', 'Z', 'RX', 'RY', 'RZ']
two_qb_gateset = ['CNOT', 'CSIGN']  
gateset = one_qb_gateset + two_qb_gateset

class EnergeticAnalysis():
    def __init__(self, jobs, results, exact_values, hardware="Superconducting", control_parameters=None):
        self.jobs = jobs
        self.results = results
        self.exact_values = exact_values
        self.hardware = hardware
        self.energetic_efficiency = []
        self.hardware_energy_consumption = []
        self.algorithmic_energy_consumption = []
        #initialization of the control parameters if not provided
        if hardware=="Superconducting":
            if control_parameters is None:
                self.control_parameters = {
                    'Tqb' : 10**(-3), #qubit temperature in K,
                    'Text' : 300, #environment temperature in K
                    'T1qb' : 50*10**(-9), #single qubit gate time in s
                    'Efac' : 1, #energetic ratio between single and two qubit gates 
                    'gamma' : 1, #spontaneous emission rate in kHz
                    'omega' : 6*10**9, #qubit frequency in Hz
                    'eta' : 0.3, #efficiency of the cryostat wrt carnot efficiency
                    'Adb' : 50 #attenuation in dB
                    }
        else:
            self.control_parameters = control_parameters
        self.run()

    @staticmethod
    def count_gates(job):
        return sum([job.circuit.count(yt) for yt in gateset])

    #hardware specific energy consumption for superconducting qubits
    def superconducting_hardware_resource(self, algo_resources):
        #insert algorithmic resources to hardware resources conversion
        #assume same energy consumption for single and two qubit gates
        #energy per gate in J
        E_1qb = hbar*self.control_parameters['omega'] * (np.pi*np.pi)/(4*self.control_parameters['gamma']*self.control_parameters['T1qb'])
        #total heat evacuated
        Adb = self.control_parameters['Adb'] #attenuation in dB   
        A = 10**(Adb/10) #attenuation in scalar
        Tqb = self.control_parameters['Tqb']
        Text = self.control_parameters['Text']
        eta = self.control_parameters['eta']
        #dressed energy in J below
        E_cool = (Text - Tqb) * A * E_1qb * algo_resources / (eta*Tqb)
        return E_cool

    def get_algorithmic_energy_consumption(self):
        """
        Returns the algorithmic energy consumption for a list of jobs
        """
        iters = len(self.jobs)
        for rt in range(iters):
            Ng = self.count_gates(self.jobs[rt]) #number of gates
            t = int(self.results[rt].meta_data['n_steps']) #number of iterations
            if self.jobs[rt].nbshots == 0:
                Ns = 1000 #default number of shots
            else:
                Ns = self.jobs[rt].nbshots
            Nm = len(self.jobs[rt].observable.terms) #number of measurements naively
            E_algo = Ng * t * Ns * Nm #algorithmic energy consumption
            self.algorithmic_energy_consumption.append(E_algo)

    def run(self):
        """
        Runs the energy consumption analysis
        """
        iters = len(self.jobs)
        self.get_algorithmic_energy_consumption()
        for rt in range(iters):
            self.hardware_energy_consumption.append(self.superconducting_hardware_resource(self.algorithmic_energy_consumption[rt]))
            err = np.abs(self.results[rt].value - self.exact_values[rt])
            self.energetic_efficiency.append(err / self.hardware_energy_consumption[rt])
        return self.energetic_efficiency, self.hardware_energy_consumption, self.algorithmic_energy_consumption