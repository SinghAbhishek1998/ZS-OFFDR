mm_cad_project/
├── requirements.txt
├── config.py
├── dataset.py
├── quantum_embedding.py  # Quantum-inspired Embedding Enhancement (Q-EE)
├── model.py              # Main MM-CAD model
└── main.py               # Training and inference loop


torch>=1.10.0
torchvision>=0.11.0
Pillow>=9.0.0
sentence-transformers>=2.2.0 # For the initial LLM-based sentence encoder
transformers>=4.20.0       # For CLIP and the main LLMs (LLaMA 2, Mistral, Gemma, Phi-2, Yi 6B)
pennylane>=0.30.0          # For differentiable quantum layer
pennylane-qiskit>=0.30.0   # PennyLane's Qiskit plugin for quantum simulation
pandas>=1.3.0
numpy>=1.20.0
scikit-learn>=1.0.0        # For F1 score calculation in evaluation
accelerate                 # For distributed training and large model handling
bitsandbytes               # For 8-bit quantization (if loading very large LLMs efficiently)