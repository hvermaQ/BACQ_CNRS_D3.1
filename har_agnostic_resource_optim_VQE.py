# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 13:54:14 2025

@author: cqtv201

features to add:
    1. test for the noiseless case before proceeding to optimization
    2. samples on the next epsilon curve drawn near the last one
"""

from qat.core import Observable, Term, Batch #Hamiltonian
from qat.qpus import get_default_qpu
from qat.fermion import SpinHamiltonian
import numpy as np
from multiprocessing import Pool
from circ_gen import gen_circ_HVA as gen_circ # import gen_circ_HVA if required
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
g_energy = eigvals[0]
g_state = eigvecs[:,0]

qpu_ideal = get_default_qpu()

# range of depth available for interpolation
dep = np.arange(1, 11, 1, dtype = int)

# range of errors: format base times 10**exponential for [base, exponential] format
base_expo = [[1,6], [5,6], [1,5], [5,5], [1,4], [5,4], [1,3], [5,3], [1,2]]
eps = [b[0]*(10**(-b[1])) for b in base_expo]

optimizer = Opto()
qpu_ideal = get_default_qpu()
# Assuming stack.submit is a method, wrapper below avoiding any object instantitation
# Wrapper function to handle multiple arguments for map
# i : dummy iterable
def submit_job(args):
    lay, nq, F, i = args  # Unpack the arguments
    circ = gen_circ(lay, nq)
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
    plt.ylabel(r"$E_\text{VQE}$")
    #plt.plot(depth_reached, found_en, '*', color = 'blue'',label = 'actual_min')
    #plt.hlines(g_en + 0.1, 1, 6, 'm', linestyles = 'dashed')
    plt.hlines(g_energy + tole, depth_all[0], depth_all[-1], 'm', linestyles = 'dashed', label = r'$\delta + %.1f$'%tole)
    plt.hlines(g_energy, depth_all[0], depth_all[-1], 'k', linestyles = 'dotted', label = r'$\delta$')
    plt.legend(bbox_to_anchor = (1,1))
    #plt.savefig("HVA_en_opt.pdf", bbox_inches = 'tight')
    plt.show()

#en_opt(nqbits, dep_list, n_samples, tol, err_rates, rds, tol_flag = False, *args)
#code to generate result. Especially required for using multiprocessing correctly
if __name__ == '__main__':
    set_tol = 0.1
    a, b, c, d, e = en_opt(nqbts, dep, 3, set_tol, eps, 5, False, g_energy)
    plotting(a, eps, dep, b, c, d, set_tol)
    if e == False:
        print("Given tolerance is viable!")
        print("Minimum error required for the given tolerance = %s"%a)
    else:
        print("Given tolerance is unviable for given error range!")
    #a, b, c, d = sampled_curve(3, -1, d_lis, 5)