import numpy as np
from scipy.optimize import minimize 
from qiskit import transpile
from qiskit_aer import AerSimulator

from feature_maps import SimpleAngleEncoder
from ansatz import SimpleAnsatz

class QMLTrainer:

    def __init__(self, num_qubits: int = 2, num_layers: int = 1):
        self.num_qubits = num_qubits
        self.num_layers = num_layers

        self.feature_map = SimpleAngleEncoder(num_qubits)
        self.ansatz = SimpleAnsatz(num_qubits, num_layers)

        self.base_circuit = self.ansatz.build_circuit()

    #params = what we are evaluating
    #X = training data
    #y = training results
    def cost_function(self, params, X, y):
        predictions = []
        for data_point in X:
            feature_circuit = self.feature_map.encode(data_point)
            full_circuit = feature_circuit.compose(self.base_circuit)
            bound_circuit = full_circuit.assign_parameters(params)
            
            # Add measurement gates
            bound_circuit.measure_all()

            simulator = AerSimulator()
            job = simulator.run(bound_circuit, shots=1024)
            result = job.result()
            counts = result.get_counts()

            prediction = self._get_prediction_from_counts(counts)
            predictions.append(prediction)

        # Step 6: Calculate cost
        correct = sum(1 for pred, true in zip(predictions, y) if pred == true)
        accuracy = correct / len(y)
        cost = 1 - accuracy

        return cost

    def _get_prediction_from_counts(self, counts):
        total_shots = sum(counts.values())
        # Calculate expectation value of first qubit
        expectation = 0
        for bitstring, count in counts.items():
            # If first bit is '0', contribute +1; if '1', contribute -1
            if bitstring[0] == '0':
                expectation += count
            else:
                expectation -= count
        expectation = expectation / total_shots
        # Convert to prediction: positive → class 0, negative → class 1
        return 0 if expectation > 0 else 1

    def train(self, X_train, y_train):
        # Get number of parameters needed
        num_params = self.ansatz.get_num_parameters()

        # Random starting point
        initial_params = np.random.uniform(0, 2*np.pi, num_params)

        # Run optimization
        result = minimize(
            fun=self.cost_function,
            x0=initial_params,
            args=(X_train, y_train),
            method='COBYLA',
            options={'maxiter': 50}
        )

        # Return results
        return {
            'optimal_params': result.x,
            'final_cost': result.fun,
            'success': result.success,
            'message': result.message
        }

