#observable sampling simple QPU, using GPU
import cupy as cp
from qat.core import Result
from qat.core.qpu import QPUHandler
from qat.core.assertion import assert_qpu
from qat.core.wrappers.result import aggregate_data
from qat.fermion import SpinHamiltonian
import numpy as np
from qat.core.wrappers.number import python_to_thrift_complex

MAX_NB_SHOTS = 1024

class GPU_QPU(QPUHandler):
    """
    def __init__(self, parameter):
        super().__init__()  # Please ensure parents constructor is called
        self._parameter = parameter
    Skeleton of a custom QPU

    This skeleton executes a circuit by running gates one by one. This skeleton also returns
    a result.
    """
    def submit_job(self, job) -> Result:
        """
        Execute a job
        The job should contain a circuit (neither an analog job nor an annealing job)

        Args:
            job: the job to execute

        Returns:
            Result: result of the computation
        """
        # Check job
        nb_shots = job.nbshots or MAX_NB_SHOTS

        assert_qpu(job.circuit is not None, "This skeleton can only execute a circuit job")
        assert_qpu(0 < nb_shots <= MAX_NB_SHOTS, "Invalid number of shots")
        #assert_qpu(job.type == ProcessingType.SAMPLE, "This QPU does not support OBSERVABLE measurement")

        # Initialize result
        result = Result()

        # Measured qubits: qubits which should be measured at the end
        # The "qubits" attribute is either:
        #   - a list of qubits (list of integers)
        #   - None (all qubits should be measured)
        measured_qubits = job.qubits or list(range(job.circuit.nbqbits))

        # Initialize the state of the quantum system as the |0⟩ state
        state = cp.zeros((2**job.circuit.nbqbits, 1), dtype=cp.complex128)
        state[0] = 1

        # Apply each gate in the circuit
        for gate_desc in job.circuit.iterate_simple():
            gate_matrix = self.gate_to_matrix(gate_desc, job.circuit.nbqbits)
            state = cp.dot(gate_matrix, state)

        # Measure the state of the specified qubits
        expectation = self.calculate_expectation(job, state)
        #print(expectation)
        # Use the Result class's internal number format
        py_complex = complex(float(expectation[0, 0].real), float(expectation[0, 0].imag))
        result._value = python_to_thrift_complex(py_complex)
        result.nbqbits = job.circuit.nbqbits
        result.parameter_map = job.parameter_map
        #print(expectation)
        # Return result
        return result

    def gate_to_matrix(self, gate_desc, nqbits):
    # Convert a gate to a matrix representation using CuPy
        gate, params, qubits = gate_desc
        if not qubits:
            raise ValueError(f"Gate {gate} has no target qubits.")
        if gate == 'H':
            return self.hadamard_matrix(qubits[0], nqbits)
        elif gate == 'X':
            return self.pauli_x_matrix(qubits[0], nqbits)
        elif gate == 'CNOT':
            if len(qubits) < 2:
                raise ValueError(f"CNOT gate requires 2 qubits, but got {len(qubits)}.")
            return self.cnot_matrix(qubits[0], qubits[1], nqbits)
        elif gate == 'RZ':
            if not params:
                raise ValueError(f"RZ gate requires a parameter, but got none.")
            return self.rz_matrix(params[0], qubits[0], nqbits)
        elif gate == 'CSIGN':
            if len(qubits) < 2:
                raise ValueError(f"CSIGN gate requires 2 qubits, but got {len(qubits)}.")
            return self.cz_matrix(qubits[0], qubits[1], nqbits)
        # Add other gates as needed
        return cp.eye(2**nqbits, dtype=cp.complex128)

    def hadamard_matrix(self, qubit, nqbits):
        # Create a Hadamard gate matrix for the specified qubit
        H = cp.array([[1, 1], [1, -1]], dtype=cp.complex128) / cp.sqrt(2)
        return self.expand_gate(H, qubit, nqbits)

    def pauli_x_matrix(self, qubit, nqbits):
        # Create a Pauli-X gate matrix for the specified qubit
        X = cp.array([[0, 1], [1, 0]], dtype=cp.complex128)
        return self.expand_gate(X, qubit, nqbits)

    def cnot_matrix(self, control, target, nqbits):
        # Create a CNOT gate matrix for the specified control and target qubits
        CNOT = cp.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1],
                        [0, 0, 1, 0]], dtype=cp.complex128)
        return self.expand_gate(CNOT, control, nqbits, target)

    def cz_matrix(self, control, target, nqbits):
        # Create a CZ (CSIGN) gate matrix for the specified control and target qubits
        CZ = cp.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, -1]], dtype=cp.complex128)
        return self.expand_gate(CZ, control, nqbits, target)

    def rz_matrix(self, theta, qubit, nqbits, *args):
        # Convert theta to GPU array
    
        if isinstance(theta, float):
            theta_gpu = cp.asarray(theta, dtype=cp.float64)
        else:
            theta_gpu = cp.random.rand(1) * 2 * np.pi
        
        #theta_gpu = cp.asarray(theta, dtype=cp.float64)
        
        # Create 2x2 RZ gate matrix directly on GPU
        RZ = cp.zeros((2, 2), dtype=cp.complex128)
        RZ[0, 0] = cp.exp(-0.5j * theta_gpu)
        RZ[1, 1] = cp.exp(0.5j * theta_gpu)
        
        return self.expand_gate(RZ, qubit, nqbits)

    def expand_gate(self, gate, qubit, nqbits, target=None):
        # Expand a single-qubit gate to a full nqbits-qubit system
        I = cp.eye(2, dtype=cp.complex128)
        if target is None:
            # Single-qubit gate
            gate_matrix = 1
            for i in range(nqbits):
                if i == qubit:
                    gate_matrix = cp.kron(gate_matrix, gate)
                else:
                    gate_matrix = cp.kron(gate_matrix, I)
        else:
            # Two-qubit gate
            gate_matrix = 1
            for i in range(nqbits):
                if i == qubit:
                    gate_matrix = cp.kron(gate_matrix, gate)
                elif i == target:
                    continue
                else:
                    gate_matrix = cp.kron(gate_matrix, I)
            # Handle non-local two-qubit gates
            if abs(qubit - target) > 1:
                gate_matrix = self.reorder_qubits(gate_matrix, qubit, target, nqbits)
        return gate_matrix

    def reorder_qubits(self, gate_matrix, qubit, target, nqbits):
        # Reorder qubits for non-local two-qubit gates
        dim = 2**nqbits
        
        # Create permutation array
        perm = []
        for i in range(dim):
            # Convert i to binary representation
            binary = format(i, f'0{nqbits}b')
            # Create list of bits
            bits = list(binary)
            # Swap the bits at qubit and target positions
            bits[qubit], bits[target] = bits[target], bits[qubit]
            # Convert back to decimal
            new_i = int(''.join(bits), 2)
            perm.append(new_i)
        
        # Create permutation matrix
        perm_matrix = cp.zeros((dim, dim), dtype=cp.complex128)
        for i in range(dim):
            perm_matrix[perm[i], i] = 1
        
        # Apply permutation to gate matrix
        return cp.dot(cp.dot(perm_matrix.T, gate_matrix), perm_matrix)

    def measure_state(self, state, measured_qubits): #TBD***
        """Measure specific qubits in the quantum state.
        
        Args:
            state: The quantum state vector
            measured_qubits: List of qubit indices to measure
        
        Returns:
            List of measurement outcomes (0 or 1) for each measured qubit
        """
        # Calculate probabilities for each basis state
        probs = cp.abs(state.flatten())**2
        
        # Initialize measurement results
        results = []
        
        # Measure each qubit
        for qubit in measured_qubits:
            # Initialize probability of measuring |1>
            prob_one = 0
            
            # Sum probabilities for all basis states where this qubit is |1>
            for i in range(len(probs)):
                # Convert state index to binary and check if qubit is 1
                if (i >> qubit) & 1:
                    prob_one += probs[i]
                    
            # Measure the qubit based on probability
            result = 1 if cp.random.random() < prob_one else 0
            results.append(result)
        
        return results

    def calculate_expectation(self, job, state):
        """Calculate expectation value of an observable.
        
        Args:
            state: The quantum state vector
            observable: Observable matrix (2^n x 2^n complex matrix)
        
        Returns:
            float: Expectation value <ψ|O|ψ>
        """
        #exact calculation for ground state

        pre_process = SpinHamiltonian(nqbits=job.observable.nbqbits, terms=job.observable.terms)
        matrix_obs = cp.array(pre_process.get_matrix(), dtype=cp.complex128)

        # Calculate <ψ|O|ψ>
        expectation = cp.dot(cp.dot(state.conj().T, matrix_obs), state)
        # Convert to Python scalar and take real part
        # This ensures compatibility with QAT's result handling
        return cp.real(expectation)