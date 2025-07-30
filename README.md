# QML-Qlassifier 🔮🤖

A Quantum Machine Learning (QML) classifier built with Qiskit that demonstrates how quantum circuits can be used for binary classification tasks. This project implements a variational quantum classifier using parameterized quantum circuits and classical optimization.

## 🌟 Features

- **Quantum Feature Encoding**: Classical data is encoded into quantum circuits using angle encoding
- **Parameterized Quantum Circuits**: Trainable quantum ansatz with rotation gates
- **Classical-Quantum Hybrid Training**: Uses classical optimization to train quantum parameters
- **Binary Classification**: Supports two-class classification problems
- **Modular Architecture**: Clean separation between feature maps, ansatz, and training logic

## 🏗️ Architecture

The QML classifier consists of three main components:

### 1. Feature Maps (`feature_maps.py`)
- **SimpleAngleEncoder**: Encodes classical data into quantum circuits using RY rotation gates
- Maps classical features to quantum states via angle encoding

### 2. Quantum Ansatz (`ansatz.py`) 
- **SimpleAnsatz**: Creates parameterized quantum circuits with trainable parameters
- Uses RY rotation gates for each qubit across multiple layers
- Provides the "learning capacity" of the quantum model

### 3. Training Engine (`train.py`)
- **QMLTrainer**: Main class that orchestrates the training process
- Implements cost function based on classification accuracy
- Uses scipy optimization (COBYLA) to find optimal parameters
- Handles prediction via quantum state measurements

## 🚀 Installation

### Prerequisites
```bash
pip install qiskit qiskit-aer numpy scipy scikit-learn
```

### Clone the Repository
```bash
git clone https://github.com/Sohamgupta1303/QML-Qlassifier.git
cd QML-Qlassifier
```

## 💡 Usage

### Basic Example

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from engine.train import QMLTrainer

# Generate sample data
X, y = make_classification(n_samples=20, n_features=2, n_classes=2, random_state=42)

# Scale data for quantum gates
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create and train the QML classifier
trainer = QMLTrainer(num_qubits=2, num_layers=1)
results = trainer.train(X_scaled, y)

print(f"Training successful: {results['success']}")
print(f"Final cost: {results['final_cost']:.3f}")
print(f"Test accuracy: {(1 - results['final_cost']):.1%}")
```

### Running the Tests

Test the feature map:
```bash
python engine/test_feature_map.py
```

Run the complete system test:
```bash
python engine/test_qml_system.py
```

## 📊 How It Works

1. **Data Encoding**: Classical features are encoded into quantum states using angle encoding (RY rotations)

2. **Parameterized Circuit**: The ansatz applies trainable RY rotations to create the quantum classifier

3. **Measurement**: The quantum circuit is measured to extract classical information

4. **Prediction**: Expectation values of measurements determine the classification

5. **Training**: Classical optimizer (COBYLA) minimizes classification error by adjusting quantum parameters

### Circuit Structure
```
|0⟩ ── RY(x₀) ── RY(θ₀) ── M
|0⟩ ── RY(x₁) ── RY(θ₁) ── M
```
Where:
- `x₀, x₁`: Classical input features  
- `θ₀, θ₁`: Trainable parameters
- `M`: Measurement operations

## 🔧 Configuration

### QMLTrainer Parameters

- `num_qubits` (int): Number of qubits (should match feature dimension)
- `num_layers` (int): Number of ansatz layers (affects model complexity)

### Training Parameters

- `method`: Optimization algorithm (default: 'COBYLA')
- `maxiter`: Maximum optimization iterations (default: 50)
- `shots`: Number of quantum circuit shots (default: 1024)

## 📈 Performance

The classifier is designed for:
- **Small datasets** (< 100 samples for reasonable training time)
- **2D feature spaces** (can be extended to higher dimensions)
- **Proof-of-concept** quantum machine learning experiments

### Typical Results
- Training time: 1-5 minutes on small datasets
- Accuracy: Often exceeds random guessing (50%) on separable data
- Best suited for linearly separable or simple non-linear problems

## 🧪 Testing

The repository includes comprehensive tests:

- `test_feature_map.py`: Tests the quantum feature encoding
- `test_qml_system.py`: End-to-end system test with synthetic data

Tests use:
- Synthetic 2D classification data
- Small sample sizes for fast execution
- Accuracy comparison against random baseline

## 🔬 Research Context

This implementation demonstrates:
- **Variational Quantum Eigensolver (VQE)** principles for optimization
- **Quantum Machine Learning** fundamentals
- **Near-term quantum algorithm** design (NISQ-era)
- **Hybrid classical-quantum** computation

## 🚧 Limitations

- **Scalability**: Limited to small datasets due to quantum simulation overhead
- **Noise**: Uses ideal quantum simulator (no noise modeling)
- **Features**: Currently supports only 2D input features
- **Optimization**: Local optimization may find suboptimal solutions

## 🔮 Future Enhancements

- [ ] Support for multi-class classification
- [ ] Noise-aware training for real quantum devices
- [ ] Advanced ansatz architectures (entangling gates)
- [ ] Gradient-based optimization methods
- [ ] Higher-dimensional feature support
- [ ] Quantum advantage analysis tools

## 📚 References

- [Variational Quantum Classifier](https://arxiv.org/abs/1804.00633)
- [Quantum Machine Learning](https://www.nature.com/articles/nature23474)
- [Qiskit Textbook](https://qiskit.org/textbook/ch-machine-learning/)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

Built with [Qiskit](https://qiskit.org/) - IBM's open-source quantum computing framework.

---

**Note**: This is an educational/research implementation. For production quantum machine learning, consider using established frameworks like PennyLane or Qiskit Machine Learning. 
