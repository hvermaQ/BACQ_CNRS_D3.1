# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 13:54:14 2025

@author: cqtv201

features to add:
    1. test for the noiseless case before proceeding to optimization
    2. samples on the next epsilon curve drawn near the last one
"""

from qat.lang.AQASM import Program, H, X, CNOT, RX, I, RY, RZ, CSIGN #Gates
from qat.core import Observable, Term, Batch #Hamiltonian
import numpy as np
#from qat.plugins import ScipyMinimizePlugin
#import pickle
from multiprocessing import Pool
import matplotlib.pyplot as plt

#gateset for counting gates to introduce noise through Gaussian noise plugin
one_qb_gateset = ['H', 'X', 'Y', 'Z', 'RX', 'RY', 'RZ']
two_qb_gateset = ['CNOT', 'CSIGN']  
#one_qb_gateset = ['H', 'RZ']
#two_qb_gateset = ['CNOT'] 
gateset = one_qb_gateset + two_qb_gateset

from qat.plugins import Junction
from qat.core import Result
from scipy.optimize import minimize
from qat.core.plugins import AbstractPlugin
from qat.fermion import SpinHamiltonian
from qat.qpus import get_default_qpu

#problem initialization
nqbts = 5 # number of qubits
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
g_energy = eigvals[0]
g_state = eigvecs[:,0]

qpu_ideal = get_default_qpu()

# range of depth available for interpolation
dep = np.arange(1, 11, 1, dtype = int)

# range of errors: format base times 10**exponential for [base, exponential] format
base_expo = [[1,6], [5,6], [1,5], [5,5], [1,4], [5,4], [1,3], [5,3], [1,2]]
eps = [b[0]*(10**(-b[1])) for b in base_expo]

#custom scipy optimization wrapper : Junction in Qaptiva

class Opto(Junction):
    def __init__(self, x0: np.ndarray = None, tol: float = 1e-8, maxiter: int = 25000, nbshots: int = 0,):
        super().__init__(collective=False)
        self.x0 = x0
        self.maxiter = maxiter
        self.nbshots = nbshots
        self.n_steps = 0
        self.energy_optimization_trace = []
        self.parameter_map = None
        self.energy = 0
        self.energy_result = Result()
        self.tol = tol
        self.c_steps = 0
        self.int_energy = []

    def run(self, job, meta_data):
        
        if self.x0 is None:
            self.x0 = 2*np.pi*np.random.rand(len(job.get_variables()))
            self.parameter_map = {name: x for (name, x) in zip(job.get_variables(), self.x0)}

        def compute_energy(x):
            job_bound =  job(** {v: xx for (v, xx) in zip(job.get_variables(), x)})
            self.energy = self.execute(job_bound)
            self.energy_optimization_trace.append(self.energy.value)
            self.n_steps += 1
            return self.energy.value

        def cback(intermediate_result):
            #fn =  compute_energy(intermediate_result)
            self.int_energy.append(intermediate_result.fun)
            self.c_steps += 1
            #return(fn)

        bnd = (0, 2*np.pi)
        bnds = tuple([bnd for i in range(len(job.get_variables()))])
        #res = minimize(compute_energy, x0 = self.x0, method='L-BFGS-B', bounds = bnds, callback = cback , options={'ftol': self.tol, 'disp': False, 'maxiter': self.maxiter})
        res = minimize(compute_energy, x0 = self.x0, method='COBYLA', bounds = bnds, options={'tol': self.tol, 'disp': False, 'maxiter': self.maxiter})
        en = res.fun
        self.parameter_map =  {v: xp for v, xp in zip(job.get_variables(), res.x)}
        self.energy_result.value = en
        self.energy_result.meta_data = {"optimization_trace": str(self.energy_optimization_trace), "n_steps": f"{self.n_steps}", "parameter_map": str(self.parameter_map), "c_steps" : f"{self.c_steps}", "int_energy": str(self.int_energy)}
        return (Result(value = self.energy_result.value, meta_data = self.energy_result.meta_data))


#custom gaussian noise plugin : Abstract plugin in qaptiva

class GaussianNoise(AbstractPlugin,):
    def __init__(self, p, hamiltonian_matrix):
        self.p = p
        self.hamiltonian_trace = np.trace(hamiltonian_matrix)/(np.shape(hamiltonian_matrix)[0])
        self.unsuccess = 0
        self.success = 0
        self.nb_pauli_strings = 0
        self.nbshots = 0
    def compile(self, batch, _):
        self.nbshots =  batch.jobs[0].nbshots
        #nb_gates = batch.jobs[0].circuit.depth({'CNOT' : 2, 'RZ' : 1, 'H' : 1}, default = 1)
        nb_gates = sum([batch.jobs[0].circuit.count(yt) for yt in gateset])
        self.success = abs((1-self.p)**nb_gates)
        self.unsuccess = (1-self.success)*self.hamiltonian_trace
        return batch 
    
    def post_process(self, batch_result):
        if batch_result.results[0].value is not None:
            for result in batch_result.results:
                if self.nbshots == 0:
                    noise =  self.unsuccess
                else: 
                    noise =  np.random.normal(self.unsuccess, self.unsuccess/np.sqrt(self.nbshots))
                result.value = self.success*result.value + noise
        return batch_result

#circuit generation using HVA
#ct = number of layers
def gen_circ_HVA(ct, nqbt):
    #Variational circuit can only be constructed using the program framework
    qprog = Program()
    qbits = qprog.qalloc(nqbt)
    #variational parameters used for generating gates (permutation of [odd/even, xx/yy/zz])
    ao = [qprog.new_var(float, 'ao_%s'%i) for i in range(ct)]
    bo = [qprog.new_var(float, 'bo_%s'%i) for i in range(ct)]
    co = [qprog.new_var(float, 'co_%s'%i) for i in range(ct)]
    ae = [qprog.new_var(float, 'ae_%s'%i) for i in range(ct)]
    be = [qprog.new_var(float, 'be_%s'%i) for i in range(ct)]
    ce = [qprog.new_var(float, 'ce_%s'%i) for i in range(ct)]
    for q_index in range(nqbt):
        X(qbits[q_index])
    for q_index in range(nqbt):
        if not q_index%2 and q_index <= nqbt-1:
            H(qbits[q_index])
    for q_index in range(nqbt):
        if not q_index%2 and q_index <= nqbt-2:
            CNOT(qbits[q_index],qbits[q_index+1])
    for it in range(ct):
        for q_index in range(nqbt): #odd Rzz
            if q_index%2 and q_index <= nqbt-2:
                CNOT(qbits[q_index],qbits[q_index+1])
                RZ(ao[it-1]/2)(qbits[q_index+1])
                CNOT(qbits[q_index],qbits[q_index+1])
        for q_index in range(nqbt): #odd Ryy
            if q_index%2 and q_index <= nqbt-2:
                RZ(np.pi/2)(qbits[q_index])
                RZ(np.pi/2)(qbits[q_index+1])
                H(qbits[q_index])
                H(qbits[q_index+1])
                CNOT(qbits[q_index],qbits[q_index+1])
                RZ(bo[it-1]/2)(qbits[q_index+1])
                CNOT(qbits[q_index],qbits[q_index+1])
                H(qbits[q_index])
                H(qbits[q_index+1])
                RZ(-np.pi/2)(qbits[q_index])
                RZ(-np.pi/2)(qbits[q_index+1])
        for q_index in range(nqbt): #odd Rxx
            if q_index%2 and q_index <= nqbt-2:
                H(qbits[q_index])
                H(qbits[q_index+1])
                CNOT(qbits[q_index],qbits[q_index+1])
                RZ(co[it-1]/2)(qbits[q_index+1])
                CNOT(qbits[q_index],qbits[q_index+1])
                H(qbits[q_index])
                H(qbits[q_index+1])
        for q_index in range(nqbt): #even Rzz
            if not q_index%2 and q_index <= nqbt-2:
                CNOT(qbits[q_index],qbits[q_index+1])
                RZ(ae[it-1]/2)(qbits[q_index+1])
                CNOT(qbits[q_index],qbits[q_index+1])
        for q_index in range(nqbt): #even Ryy
            if not q_index%2 and q_index <= nqbt-2:
                RZ(np.pi/2)(qbits[q_index])
                RZ(np.pi/2)(qbits[q_index+1])
                H(qbits[q_index])
                H(qbits[q_index+1])
                CNOT(qbits[q_index],qbits[q_index+1])
                RZ(be[it-1]/2)(qbits[q_index+1])
                CNOT(qbits[q_index],qbits[q_index+1])
                H(qbits[q_index])
                H(qbits[q_index+1])
                RZ(-np.pi/2)(qbits[q_index])
                RZ(-np.pi/2)(qbits[q_index+1])
        for q_index in range(nqbt): #even Rxx
            if not q_index%2 and q_index <= nqbt-2:
                H(qbits[q_index])
                H(qbits[q_index+1])
                CNOT(qbits[q_index],qbits[q_index+1])
                RZ(ce[it-1]/2)(qbits[q_index+1])
                CNOT(qbits[q_index],qbits[q_index+1])
                H(qbits[q_index])
                H(qbits[q_index+1])
    #circuit
    circuit = qprog.to_circ()
    return(circuit)

#circuit generation for RYA ansatz
#ct is the depth

def gen_circ_RYA(ct, nqbt):
    #Variational circuit can only be constructed using the program framework
    qprog = Program()
    qbits = qprog.qalloc(nqbt)
    #variational parameters used for generating gates (permutation of [odd/even, xx/yy/zz])
    #variational parameters used for generating gates
    angs = [qprog.new_var(float, 'a_%s'%i) for i in range(nqbt*(2*ct+1))]
    ang = iter(angs)
    #circuit
    for it in range(ct):
        for q_index in range(nqbt):
            RY(next(ang))(qbits[q_index])
        for q_index in range(nqbt):
            if not q_index%2 and q_index <= nqbt-2:
                CSIGN(qbits[q_index],qbits[q_index+1])
        for q_index in range(nqbt):
            RY(next(ang))(qbits[q_index])
        for q_index in range(nqbt):
            if q_index%2 and q_index <= nqbt-2:
                CSIGN(qbits[q_index],qbits[q_index+1])
        CSIGN(qbits[0],qbits[nqbt-1])
        if it==(ct-1):
            for q_index in range(nqbt):
                RY(next(ang))(qbits[q_index])
    #circuit
    circuit = qprog.to_circ()
    return(circuit)

optimizer = Opto()
qpu_ideal = get_default_qpu()
# Assuming stack.submit is a method, wrapper below avoiding any object instantitation
# Wrapper function to handle multiple arguments for map
# i : dummy iterable
def submit_job(args):
    lay, nq, F, i = args  # Unpack the arguments
    circ = gen_circ_HVA(lay, nq)
    if F < 0:         
        stack = optimizer | qpu_ideal  # ideal stack
    else:
        stack = optimizer | GaussianNoise(F, heisen_mat) | qpu_ideal  # noisy stack
    job = circ.to_job(observable=heisen, nbshots=0)
    result = stack.submit(job).value
    return result

# Simple program running wrapper: VQE stack
# reverts to ideal run if F<0
# F: infidelity, layers: relevant layer list
def run_prog_map(layers, qubits, Fidelity, randoms):
    # Parallel run handled here
    with Pool() as p:  # Number of processes can be adjusted
        result_async = p.map_async(submit_job, [(layer, qubits, Fidelity, y) for layer in layers for y in range(randoms)])  # Passing arguments as a tuple
        #ress = p.map(submit_job, [(layers, qubits, Fidelity, y) for y in range(randoms)])
        # Get the results from the asynchronous map
        ress = result_async.get()  # This will block until all results are available
    # Now, we want to extract the minimum result for each layer
    min_results_per_layer = []
    # Iterate through each layer and find the minimum result for each layer
    for i, layer in enumerate(layers):
        layer_results = ress[i * randoms: (i + 1) * randoms]  # Extract the results for the current layer
        min_results_per_layer.append(np.min(layer_results))  # Find the minimum result for this layer
    return(min_results_per_layer)  # Return the minimum result

# curve sampling routine
def sampled_curve(nqbits, eps_err, d_list, rnds):
    #res = []
    #step 1: run for d_list, with random seeds || all parallel
    result = run_prog_map(d_list, nqbits, eps_err, rnds)
    #res.append(result)
    #step 2: fit curves
    #fit exponential if eps_err < 0 (proxy for ideal)
    if (eps_err < 0):
        pos = np.argmin(result)  # Use last point as initial guess supposedly exponential decay
        proj_min = np.min(result)
        ps = 0
    # Fit as usual a polynomial of degree: n_samples - 1 if eps_err > 0
    else:
        # Curve fitting routine, degree of polynomial = n_samples - 1
        ps = np.polyfit(d_list, result, len(d_list) - 1)
        # poly = np.polynomial.Polynomial(ps[::-1])
        poly_vals = np.polyval(ps, dep)
        pos = int(np.argmin(poly_vals))
        # Check if pos is outside the bounds of d_list, if yes revert to the median
        if pos < 0:
            pos = 0
        elif pos >= len(d_list):
            pos = len(d_list)-1
        # Step 3: find the exact answer at projected minima
        proj_min = run_prog_map([d_list[pos]], nqbits, eps_err, rnds)
    id_depth = d_list[pos]
    #res: sampled results, proj_min: projected minima, id_depth: depth, ps: fitting coefficients
    return(result, proj_min, id_depth, ps)

#energy optimization plugin starts here
#note that base expo should be in ascending order

#inputs: self explanatory, except -> n_samples : number of points to sample on a curve, tol : tolerance level for metric/ accuracy gain per curve switching
#inputs: tol_flag: Flase if exact answer is known for comparison, True if accuracy gain per curve switch is required
#args takes the target metric only (optional ; required only if tol_flag = False)

#returns eps: the last value of error/ error rate for which the metric satisfies the tol condition
#minima : series of calculated values at projected minima throughout the routine run
#ideal depth : series of number of layers at the projected minima
#raw data : polynomial coefficients for curve fits
#flag: True if not convereged
def en_opt(nqbits, dep_list, n_samples, tol, err_rates, rds, tol_flag = False, *args):
    d_lis = dep_list[::len(dep_list)//n_samples] #prepared list of sample depths
    vals = []
    ideal_depth = []
    minima = []
    raw_data = []
    #noiseless run for comparison with the target only if tol_flag is False
    if not tol_flag:
        a_ideal, b_ideal, c_ideal, d_ideal = sampled_curve(nqbits, -1, d_lis, rds)
        vals.append(a_ideal)
        minima.append(b_ideal)
        ideal_depth.append(c_ideal)
        raw_data.append(d_ideal)
        # Check if projected minima in the ideal case exceeds the target
        if (b_ideal - args[0] > tol):
            print("Required accuracy cannot be achieved: projected ideal minima exceeds tolerance.")
            return None  # Terminate early with no further calculations
    maxx = len(err_rates)
    a, b, c, d = sampled_curve(nqbits, err_rates[-1], d_lis, rds)
    vals.append(a)
    minima.append(b)
    ideal_depth.append(c)
    raw_data.append(d)
    init = 2 # dummy initial for descending through the energy/error list
    tol_obs = 1 # initialization for comparison, also dummy
    count = 1
    flag = 0
    if c!=1:
        d_lis = np.arange(c - (n_samples-1)//2, c + n_samples-(n_samples-1)//2, dtype = int)
    else:
        d_lis = np.arange(1, 1 + n_samples, dtype = int)
    if tol_flag:
        while (tol_obs > tol) and (count<=maxx):
            a, b, c, d = sampled_curve(nqbits, err_rates[-init], d_lis, rds)
            vals.append(a)
            minima.append(b)
            ideal_depth.append(c)
            raw_data.append(d)
            init+=1
            tol_obs = np.abs(minima[-1] - minima[-2])
            count+=1
            #switching to next curve with the sampled points concentrated towards projected minima from the previous curve
            if c!=1:
                d_lis = range(c - (n_samples-1)//2, c + n_samples-(n_samples-1)//2)
            else:
                d_lis = range(1, 1 + n_samples)
    else:
        while (np.abs(b - args[0]) >= tol) and (count<=maxx):
            a, b, c, d = sampled_curve(nqbits, err_rates[-init], d_lis, rds)
            vals.append(a)
            minima.append(b)
            ideal_depth.append(c)
            raw_data.append(d)
            init+=1
            count+=1
            #switching to next curve with the sampled points concentrated towards projected minima from the previous curve
            if c!=1:
                d_lis = range(c - (n_samples-1)//2, c + n_samples-(n_samples-1)//2)
            else:
                d_lis = range(1, 1 + n_samples)
    if count == maxx:
         eps = 0
    else:
        eps = err_rates[-init+1]
    if (np.abs(b - args[0]) > tol):
        flag = True
    else:
        flag = False
    return(eps, minima, ideal_depth, raw_data, flag)

def plotting(eps_reached, eps_all, depth_all, obtained_minima, minima_depth, fitting_coeff, tole):
    #plot the interpolated plots with polynomial
    #point and plot projected minima
    
    fit_coeff = [fitting_coeff[0]] + fitting_coeff[1:][::-1]
    min_depth = [minima_depth[0]] + minima_depth[1:][::-1]
    min_value = [obtained_minima[0]] + obtained_minima[1:][::-1]
    
    # Ensure min_value and min_depth are one-dimensional and contain scalars, not lists
    min_value = [item[0] if isinstance(item, list) else item for item in obtained_minima]
    min_depth = [item[0] if isinstance(item, list) else item for item in minima_depth]
    
    # Find the index of the first occurrence of the value
    index = np.where(np.array(eps_all) == eps_reached)[0][0]  # Get the first index where the value occurs
    # Slice the array up to the value
    eps_value = [0] + eps_all[index:]
    #data arranged as increasing in noise starting with noiseless
    
    for bt in range(len(min_value)):        
        if bt==0:
            plt.hlines(min_value[0], depth_all[0], depth_all[-1], 'c', linestyles = 'dashdot', label = r'$\epsilon = 0$')
        else:
            tt = np.polynomial.Polynomial(fit_coeff[bt][::-1])
            vals = tt(depth_all)  # Polynomial evaluation over a range
            # Format eps_value[bt] as x × 10^y
            eps_str = "{:.1e}".format(eps_value[bt])  # Convert to scientific notation
            base, exponent = eps_str.split("e")  # Split into base and exponent
            base = float(base)  # Convert the base part to a float
            exponent = int(exponent)  # Convert the exponent part to an integer
            plt.plot(depth_all, vals, label=r"$ \epsilon = %.1f \times 10^{%d}$" % (base, exponent))
 
        #min_vals.append(np.min(tt(np.linspace(1,6,6))))
        #pos.append(np.argmin(tt(np.linspace(1,6,6))))
    
    # Use scatter to add special markers for min_depth and min_value
    plt.scatter(min_depth, min_value, color='red', label="Minima Points: actual", zorder=5, marker='o')
    plt.ylim(-8, 5)

    #plt.yscale('log')
    plt.xlabel(r"$N_{layers}$")
    plt.ylabel(r"$E_\infty$")
    #plt.plot(depth_reached, found_en, '*', color = 'blue'',label = 'actual_min')
    #plt.hlines(g_en + 0.1, 1, 6, 'm', linestyles = 'dashed')
    plt.hlines(g_energy + tole, depth_all[0], depth_all[-1], 'm', linestyles = 'dashed', label = 'G.S +0.1')
    plt.hlines(g_energy, depth_all[0], depth_all[-1], 'k', linestyles = 'dotted', label = 'G.S')
    plt.legend(bbox_to_anchor = (1,1))
    #plt.savefig("HVA_en_opt.pdf", bbox_inches = 'tight')
    plt.show()

#en_opt(nqbits, dep_list, n_samples, tol, err_rates, rds, tol_flag = False, *args)
#code to generate result. Especially required for using multiprocessing correctly
if __name__ == '__main__':
    a, b, c, d, e = en_opt(nqbts, dep, 3, 0.1, eps, 5, False, g_energy)
    plotting(a, eps, dep, b, c, d, 0.1)
    if e == False:
        print("Given tolerance is viable!")
        print("Minimum error required for the given tolerance = %s"%a)
    else:
        print("Given tolerance is unviable for given error range!")
    #a, b, c, d = sampled_curve(3, -1, d_lis, 5)