#energy and efficiency estimator based on results from VQE
# -*- coding: utf-8 -*-

from qat.core import Observable, Term, Batch #import for Hamiltonian Contruction in myqlm
from qat.qpus import get_default_qpu #deafult qpu for vqe
from qat.fermion import SpinHamiltonian #additional import for Hamiltonian construction
import numpy as np #numpy for calculations
#from multiprocessing import Pool #optional for parallelization for large number of jobs
from circ_gen import gen_circ_HVA as gen_circ # circuit generation function, can switch to other ansatz

from opto_gauss import Opto, GaussianNoise #custom noise model and optimizer for noisy VQE simulation

#problem initialization
nqbts = 3 # number of qubits
nruns = 0 # nbshots for observable sampling

#Instantiation of Hamiltoniian
heisen = Observable(nqbts)
#Generation of Heisenberg Hamiltonian
for q_reg in range(nqbts-1):
    heisen += Observable(nqbts, pauli_terms = [Term(1., typ, [q_reg,q_reg + 1]) for typ in ['XX','YY','ZZ']])  

#exact calculation for ground state

heisen_class = SpinHamiltonian(nqbits=heisen.nbqbits, terms=heisen.terms)
heisen_mat = heisen_class.get_matrix()

eigvals, eigvecs = np.linalg.eigh(heisen_mat)
#ground state energy
g_energy = eigvals[0] #ground state energy
g_state = eigvecs[:,0] #ground state vector (optional)

# range of depth available for ansatz
depth = 4

# error rate in error model
eps = 5*10**(-4)

#tools initialization
optimizer = Opto()
qpu_ideal = get_default_qpu()


circ = gen_circ(depth, nqbts)

stack = optimizer | GaussianNoise(eps, heisen_mat) | qpu_ideal  # noisy stack
jobb = circ.to_job(observable=heisen, nbshots=nruns)
result = stack.submit(jobb)

#importing the EnergeticAnalysis class
from energetics import EnergeticAnalysis
#requires the ordered list of jobs, results and exact values
#single job considered here for illustration
ress = EnergeticAnalysis([jobb], [result], [g_energy], hardware="Superconducting", control_parameters=None)
print("Algorithmic energy consumption in abstract units: ", ress.algorithmic_energy_consumption)
print("Hardware energy consumption in Joules: ", ress.hardware_energy_consumption)
print("Hardware Energetic efficiency = error/ hardware energy consumption: ", ress.energetic_efficiency)