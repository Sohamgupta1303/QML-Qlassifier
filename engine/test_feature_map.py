import numpy as np
from feature_maps import SimpleAngleEncoder

# Test your encoder
encoder = SimpleAngleEncoder(num_qubits=2)
test_data = np.array([0.5, 0.8])
circuit = encoder.encode(test_data)

print("Your quantum circuit:")
print(circuit)
