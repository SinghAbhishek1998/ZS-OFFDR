import os
import torch

# --- General Configuration ---
PROJECT_NAME = "MM-CAD"
DATA_DIR = "path/to/your/DACAM_dataset" # <<< !!! IMPORTANT: REPLACE WITH YOUR ACTUAL DACAM DATASET PATH !!!
# Example: If DACAM has a CSV describing memes and an image folder
CSV_FILE = os.path.join(DATA_DIR, "dacam_annotations.csv")
IMG_DIR = os.path.join(DATA_DIR, "images")
LOG_DIR = "logs"
RESULTS_DIR = "results"
CHECKPOINT_DIR = "checkpoints"

# --- Model Configuration ---
# Visual Encoder
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# Textual Encoder (LLM-based sentence encoder)
# This will be used to get initial text embeddings, which are then passed to alignment.
# As per methodology, it's a "lightweight sentence encoder from open-source LLMs".
# SentenceTransformer("intfloat/multilingual-e5-base") is a good example here.
# For fine-tuning LLaMA 2, Mistral directly for embeddings, you'd load their base models.
TEXT_EMBEDDER_MODEL_NAME = "intfloat/multilingual-e5-base" # Example: A compact text embedder

# Stage 1 - Abusive Meme Classification (LLM_class)
# This LLM will have a classification head on top.
LLM_CLASSIFIER_MODEL_NAME = "mistralai/Mistral-7B-v0.1" # Example: Mistral-7B base model for classification

# Stage 2 - Explanation Generation (LLM_explain)
# This LLM should be an instruction-tuned model.
LLM_EXPLAINER_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1" # Example: Mistral-7B Instruct

# --- Embedding Alignment Configuration ---
SHARED_LATENT_DIM = 512 # 'd' in methodology for projected V' and T'
FF_DIM_MULTIPLIER = 4 # For Transformer feed-forward expansion in cross-attention
NUM_ATTENTION_HEADS = 8
NUM_ATTENTION_LAYERS = 2 # Number of bidirectional cross-attention layers

# --- Quantum Embedding Configuration (Q-EE) ---
# num_qubits: Should ideally be related to the SHARED_LATENT_DIM or a reduced representation.
# If SHARED_LATENT_DIM is 512, mapping to 8-16 qubits is common in QML experiments.
Q_NUM_QUBITS = 12 # Must be <= SHARED_LATENT_DIM
Q_NUM_LAYERS = 2 # Number of PQC layers (reps)

# --- Training Configuration ---
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_EPOCHS = 10
MAX_SEQ_LENGTH = 128 # Max token length for text input to LLM-based encoder
MAX_EXPLANATION_LENGTH = 256 # Max tokens for generated explanations

# --- Hardware Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Ensure directories exist ---
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --- !!! ETHICAL CONSIDERATION & DISCLAIMER !!! ---
# This code blueprint is for research understanding of MM-CAD.
# It deals with highly sensitive data (child abuse detection).
# ANY USE, DEPLOYMENT, OR FURTHER DEVELOPMENT MUST BE CONDUCTED WITH:
# 1. Strict adherence to ethical guidelines and local/international laws.
# 2. Comprehensive human oversight and expert validation.
# 3. Robust bias detection and mitigation strategies.
# 4. Strong privacy-preserving measures for data.
# This code is NOT intended for production use without extensive Responsible AI practices.
