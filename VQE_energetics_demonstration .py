#energy and efficiency estimator based on results from VQE
# -*- coding: utf-8 -*-

from qat.core import Observable, Term, Batch #Hamiltonian
from qat.qpus import get_default_qpu
from qat.fermion import SpinHamiltonian
import numpy as np
from multiprocessing import Pool
from circ_gen import gen_circ_HVA as gen_circ # import gen_circ_RYA if required
import matplotlib.pyplot as plt

from opto_gauss import Opto, GaussianNoise

plt.rcParams.update({'font.size': 16})  # Set global font size

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
g_energy = eigvals[0]
g_state = eigvecs[:,0]

# range of depth available for interpolation
depth = 4

# error rate in error model
eps = 5*10**(-4)

#tools initialization
optimizer = Opto()
qpu_ideal = get_default_qpu()


circ = gen_circ(depth, nqbts)

stack = optimizer | GaussianNoise(eps, heisen_mat) | qpu_ideal  # noisy stack
jobb = circ.to_job(observable=heisen, nbshots=0)
result = stack.submit(jobb)

from energetics import EnergeticAnalysis
ress = EnergeticAnalysis([jobb], [result], [g_energy], hardware="Superconducting")