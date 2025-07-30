import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from train import QMLTrainer

print("🧪 Testing QML Prototyper System")
print("=" * 40)

# Step 1: Generate test data
print("📊 Generating test dataset...")
X, y = make_classification(
    n_samples=20,       # Small dataset for fast testing
    n_features=2,       # 2D data for 2 qubits
    n_redundant=0,
    n_informative=2,
    n_clusters_per_class=1,
    class_sep=1.5,      # Make it easier to classify
    random_state=42
)

# Scale data to reasonable range for quantum gates
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Classes: {np.unique(y)}")

# Step 2: Create and test trainer
print("\n🔧 Creating QML Trainer...")
trainer = QMLTrainer(num_qubits=2, num_layers=1)

print(f"Number of parameters: {trainer.ansatz.get_num_parameters()}")

# Step 3: Test individual components
print("\n🧠 Testing individual components...")

# Test feature map
print("Testing feature map...")
test_data = X_train[0]
feature_circuit = trainer.feature_map.encode(test_data)
print(f"Feature circuit depth: {feature_circuit.depth()}")

# Test ansatz
print("Testing ansatz...")
ansatz_circuit = trainer.ansatz.build_circuit()
print(f"Ansatz circuit depth: {ansatz_circuit.depth()}")
print(f"Ansatz parameters: {list(ansatz_circuit.parameters)}")

# Test cost function with random parameters
print("\nTesting cost function...")
random_params = np.random.uniform(0, 2*np.pi, trainer.ansatz.get_num_parameters())
cost = trainer.cost_function(random_params, X_train[:3], y_train[:3])  # Test on small subset
print(f"Random parameters cost: {cost:.3f}")

# Step 4: Run training
print("\n🚀 Starting training...")
print("This may take a minute...")

try:
    results = trainer.train(X_train, y_train)
    
    print(f"\n✅ Training completed!")
    print(f"Success: {results['success']}")
    print(f"Final cost: {results['final_cost']:.3f}")
    print(f"Optimal parameters: {results['optimal_params']}")
    
    # Test trained model
    print("\n📊 Testing trained model...")
    trained_cost = trainer.cost_function(
        results['optimal_params'], 
        X_test, 
        y_test
    )
    
    test_accuracy = 1 - trained_cost
    print(f"Test accuracy: {test_accuracy:.1%}")
    
    # Compare to random guessing
    random_accuracy = 0.5  # 50% for binary classification
    print(f"Random guessing: {random_accuracy:.1%}")
    
    if test_accuracy > random_accuracy:
        print("🎉 SUCCESS: Model learned something!")
    else:
        print("⚠️  WARNING: Model didn't learn much...")
        
except Exception as e:
    print(f"❌ Error during training: {e}")
    import traceback
    traceback.print_exc()

print("\n�� Test complete!") 