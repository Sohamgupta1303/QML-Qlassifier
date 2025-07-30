import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

class SimpleAnsatz: 
    def __init__(self, num_qubits: int, num_layers: int = 1):
        self.num_qubits = num_qubits
        self.num_layers = num_layers

    def get_num_parameters(self) -> int: 
        return self.num_layers * self.num_qubits
    
    def build_circuit(self) -> QuantumCircuit:
        num_params = self.get_num_parameters()
        params = ParameterVector('Θ', num_params)

        circuit = QuantumCircuit(self.num_qubits)

        param_index = 0
        for layer in range(self.num_layers):
            for qubit in range(self.num_qubits):
                circuit.ry(params[param_index], qubit)
                param_index += 1

        return circuit