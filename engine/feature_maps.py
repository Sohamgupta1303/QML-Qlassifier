import numpy as np
from qiskit import QuantumCircuit

class SimpleAngleEncoder:
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits

    def encode(self, data: np.ndarray) -> QuantumCircuit:
        circuit = QuantumCircuit(self.num_qubits)
        for i, angle in enumerate(data[:self.num_qubits]):
            circuit.ry(angle, i)

        return circuit