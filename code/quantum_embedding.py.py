import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as np # Pennylane's wrapped NumPy for differentiability
import config

class QuantumEmbeddingLayer(nn.Module):
    def __init__(self, classical_input_dim, num_qubits, num_layers=1):
        """
        Initializes the Quantum-Enhanced Embedding (Q-EE) layer.
        Args:
            classical_input_dim (int): Dimension of the classical aligned multimodal embedding (z_i).
            num_qubits (int): Number of qubits to use in the PQC.
                              Must be <= classical_input_dim.
            num_layers (int): Number of layers (reps) in the PQC.
        """
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.classical_input_dim = classical_input_dim

        if num_qubits > classical_input_dim:
            raise ValueError(f"num_qubits ({num_qubits}) cannot be greater than classical_input_dim ({classical_input_dim}) for angle embedding.")

        # Classical preprocessing to match input dim to num_qubits for angle embedding
        self.input_projection = nn.Linear(classical_input_dim, num_qubits)

        # Define the PennyLane device
        # 'default.qubit' is a local simulator. For faster simulation (e.g., GPU), consider 'lightning.qubit'.
        # For actual quantum hardware, this would be a different device setup.
        self.dev = qml.device("default.qubit", wires=self.num_qubits, shots=None) # shots=None for exact expectation values

        # Define the QNode (quantum circuit)
        @qml.qnode(self.dev, interface="torch", diff_method="backprop") # diff_method="backprop" for simulator-based backprop
        def quantum_circuit(inputs, weights):
            # Step 1: Feature map (phi function conceptually)
            # AngleEmbedding maps classical input vector onto rotation angles for qubits.
            qml.AngleEmbedding(inputs, wires=range(self.num_qubits))

            # Step 2: Variational Unitary Transformation (U function conceptually)
            # Using StronglyEntanglingLayers as a common trainable circuit structure
            # This implements the alternating layers of single-qubit rotations and entangling gates
            qml.StronglyEntanglingLayers(weights, wires=range(self.num_qubits))

            # Step 3: Measurement (rho function conceptually)
            # Return expectation values of PauliZ on each qubit as the quantum feature vector
            return [qml.expval(qml.PauliZ(w)) for w in range(self.num_qubits)]

        self.quantum_circuit = quantum_circuit

        # Initialize trainable parameters for the variational circuit (U)
        # qml.templates.StronglyEntanglingLayers.shape gives the correct shape for the weights
        weight_shape = qml.templates.StronglyEntanglingLayers.shape(
            n_wires=self.num_qubits, n_layers=self.num_layers
        )
        self.weights = nn.Parameter(torch.rand(weight_shape) * np.pi) # Initialize with random angles

        # Post-quantum classical layer to project quantum features to desired output dimension
        # If your methodology says q_i in C^2d and it's 2d classical, then the output dim here might be different.
        # Assuming final output is real-valued vector of same dimension as input classical embedding.
        self.post_quantum_projection = nn.Linear(self.num_qubits, classical_input_dim)


    def forward(self, z_i):
        # 1. Classical projection/reduction for quantum input
        z_i_proj_for_quantum = self.input_projection(z_i) # (batch_size, num_qubits)

        # 2. Convert to Pennylane's NumPy and detach for QNode input
        # Pennylane needs a NumPy array (its own wrapped np) for qnode execution if interface is not directly torch_interface for batching.
        # For torch interface, qml.to_torch can handle tensor directly, but ensure correct device.
        # Direct batching in QNode is also possible for performance.
        
        # For simple demonstration, iterate over batch and stack results.
        # NOTE: Iterating over batches like this is slow for large batches on CPU simulators.
        # For optimal performance, consider:
        # a) Using a faster device (e.g., qml.device("lightning.qubit")).
        # b) Vectorizing the QNode execution if your PennyLane version supports it with your chosen device.
        # c) Batching inputs within the QNode using qml.vmap or similar transformations.
        
        quantum_features_list = []
        for i in range(z_i_proj_for_quantum.shape[0]): # Iterate through batch
            # Ensure input for QNode is a 1D tensor/numpy array for single circuit run
            single_input = z_i_proj_for_quantum[i].float().cpu().detach().numpy()
            
            # Execute the quantum circuit
            exp_vals = self.quantum_circuit(single_input, self.weights)
            
            # Convert Pennylane result back to PyTorch tensor and move to original device
            quantum_features_list.append(torch.tensor(exp_vals, dtype=torch.float32).to(z_i.device))
        
        quantum_features_batch = torch.stack(quantum_features_list) # (batch_size, num_qubits)

        # 3. Post-quantum classical projection
        q_i = self.post_quantum_projection(quantum_features_batch) # (batch_size, classical_input_dim)

        return q_i


# --- Example Usage (Conceptual) ---
if __name__ == '__main__':
    import config
    
    # Simulate a classical aligned multimodal embedding (z_i)
    batch_size = 4
    aligned_multimodal_embedding_dim = config.SHARED_LATENT_DIM * 2 # As per methodology (2d)
    z_i_example = torch.randn(batch_size, aligned_multimodal_embedding_dim).to(config.DEVICE)

    # Initialize Quantum Embedding Layer
    q_layer = QuantumEmbeddingLayer(
        classical_input_dim=aligned_multimodal_embedding_dim,
        num_qubits=config.Q_NUM_QUBITS, # e.g., 12 qubits
        num_layers=config.Q_NUM_LAYERS
    ).to(config.DEVICE)

    # Forward pass
    q_i_output = q_layer(z_i_example)
    print(f"Input z_i shape: {z_i_example.shape}")
    print(f"Output q_i shape: {q_i_output.shape}")

    # Test differentiability (conceptual)
    loss = q_i_output.mean()
    loss.backward()
    print("Gradient for quantum layer weights (should not be None):", q_layer.weights.grad is not None)
    if q_layer.weights.grad is not None:
        print("Quantum layer weights gradient shape:", q_layer.weights.grad.shape)

    # !!! IMPORTANT !!!
    # This example iterates over the batch on a local simulator which is slow.
    # For actual training, ensure PennyLane is set up for vectorized execution or a faster device.