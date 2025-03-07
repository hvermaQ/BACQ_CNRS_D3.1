#custom scipy optimization wrapper : Junction in Qaptiva
from qat.plugins import Junction
from scipy.optimize import minimize
from qat.core.plugins import AbstractPlugin
import numpy as np
from qat.core import Result

#gateset for counting gates to introduce noise through Gaussian noise plugin
one_qb_gateset = ['H', 'X', 'Y', 'Z', 'RX', 'RY', 'RZ']
two_qb_gateset = ['CNOT', 'CSIGN']  
gateset = one_qb_gateset + two_qb_gateset

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
