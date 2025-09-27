import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from tqdm import tqdm
import os
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
import time # For timing operations

from config import *
from dataset import MMCAD_Dataset
from model import MMCAD_Model

# --- !!! ETHICAL CONSIDERATION & DISCLAIMER !!! ---
# This code handles highly sensitive data. Refer to config.py for detailed warnings.
# Ensure all uses comply with ethical guidelines and privacy regulations.
# NEVER deploy this model in production without extensive human validation and expert review.

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        pixel_values = batch['pixel_values'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        
        logits = model(pixel_values, input_ids, attention_mask)

        loss = criterion(logits, labels)
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        
        progress_bar.set_postfix({"loss": loss.item()})
    return total_loss / len(dataloader)

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating")
        for batch in progress_bar:
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(pixel_values, input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            predictions = torch.argmax(logits, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            progress_bar.set_postfix({"loss": loss.item()})

    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='binary') # For binary classification
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, accuracy, f1

def generate_explanations(classifier_model, explainer_pipeline, dataset_subset, device):
    classifier_model.eval()
    generated_explanations = []

    print(f"\nGenerating explanations for {len(dataset_subset)} samples (only for abusive cases):")
    for i, sample_data in enumerate(tqdm(dataset_subset, desc="Generating Explanations")):
        # Unsqueeze adds a batch dimension of 1 for single sample processing
        pixel_values = sample_data['pixel_values'].unsqueeze(0).to(device)
        input_ids = sample_data['input_ids'].unsqueeze(0).to(device)
        attention_mask = sample_data['attention_mask'].unsqueeze(0).to(device)
        raw_text_content = sample_data['raw_text_content'] # This is a string, no need to unsqueeze

        with torch.no_grad():
            # Stage 1: Classification
            logits = classifier_model(pixel_values, input_ids, attention_mask)
            predicted_label = torch.argmax(logits, dim=1).item()

            explanation_text = "N/A (Not abusive)"
            if predicted_label == 1: # If classified as abusive
                # Stage 2: Explanation Generation Prompt (without RAG component)
                # The prompt structure directly uses the meme's content and classification
                
                # You might need to derive features from q_i for the prompt if LLM_explain takes q_i directly.
                # However, your methodology says LLM_explain(Prompt(q_i, y_i)), implying prompt contains q_i info.
                # For practical LLMs like Mistral-Instruct, they usually take text prompts.
                # So we'll prompt based on the raw_text_content and the abusive classification.
                
                prompt_template = f"""
                Task: Generate a concise explanation for why the following meme is classified as abusive.
                Meme Text: "{raw_text_content}"
                
                Explanation: This meme is abusive because:
                """
                
                try:
                    # The pipeline returns a list of dictionaries, take the first one's generated_text
                    generated_response = explainer_pipeline(
                        prompt_template,
                        max_new_tokens=MAX_EXPLANATION_LENGTH,
                        do_sample=True,
                        temperature=0.7,
                        top_k=50,
                        top_p=0.95,
                        num_return_sequences=1,
                        return_full_text=False # Only return the new generated text
                    )
                    explanation_text = generated_response[0]['generated_text'].strip()
                except Exception as e:
                    explanation_text = f"Error generating explanation: {e}"
                    print(f"Error during explanation generation for '{raw_text_content}': {e}")
            
            generated_explanations.append({
                'meme_text': raw_text_content,
                'true_label': sample_data['labels'].item(),
                'predicted_label': predicted_label,
                'explanation': explanation_text
            })
    return generated_explanations

def main():
    # --- !!! ETHICAL CONSIDERATION !!! ---
    print("!!! WARNING: This code handles sensitive data related to child abuse detection. !!!")
    print("!!! Ensure all uses comply with ethical guidelines and privacy regulations.     !!!")
    print("!!! NEVER deploy this model in production without extensive human validation.  !!!")
    print("-" * 80)

    # 1. Load Tokenizer and Dataset
    print(f"Loading tokenizer for {LLM_CLASSIFIER_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_CLASSIFIER_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

    print("Loading DACAM dataset (conceptual)...")
    # For actual DACAM, ensure your config.CSV_FILE points to the split files.
    # If DACAM is a single CSV, split it into train/val/test first.
    train_dataset = MMCAD_Dataset(
        csv_file=CSV_FILE, # Placeholder for your actual train CSV
        img_dir=IMG_DIR,
        clip_processor=CLIPProcessor.from_pretrained(CLIP_MODEL_NAME),
        text_tokenizer=tokenizer,
        max_seq_length=MAX_SEQ_LENGTH
    )
    # Re-use for val/test or create separate CSVs if your DACAM has them
    val_dataset = train_dataset # For demo, use same data
    test_dataset = train_dataset # For demo, use same data

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # 2. Initialize Model, Optimizer, Loss
    print("Initializing MMCAD_Model...")
    model = MMCAD_Model().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # 3. Training Loop (Stage 1: Abusive Meme Classification)
    print("\nStarting Stage 1: Abusive Meme Classification Training...")
    best_f1 = 0
    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_accuracy, val_f1 = evaluate_model(model, val_loader, criterion, DEVICE)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"{PROJECT_NAME}_best_classifier.pth"))
            print("Saved best model checkpoint.")
    
    # Load best model for final test evaluation or explanation generation
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, f"{PROJECT_NAME}_best_classifier.pth")))
    print("\nLoaded best classifier model.")

    # 4. Final Evaluation on Test Set
    print("\nPerforming final evaluation on test set...")
    test_loss, test_accuracy, test_f1 = evaluate_model(model, test_loader, criterion, DEVICE)
    print(f"Final Test Results - Loss: {test_loss:.4f}, Acc: {test_accuracy:.4f}, F1: {test_f1:.4f}")

    # 5. Stage 2: Explanation Generation (on a subset of test data)
    print("\nStarting Stage 2: Explanation Generation...")
    
    # Load the LLM for explanation generation using Hugging Face pipeline
    print(f"Loading LLM for explanation generation: {LLM_EXPLAINER_MODEL_NAME}...")
    try:
        explainer_pipeline = pipeline(
            "text-generation",
            model=LLM_EXPLAINER_MODEL_NAME,
            torch_dtype=torch.float16, # Use float16 for memory efficiency
            device_map="auto",         # Distribute model across available GPUs
            tokenizer=tokenizer        # Use the same tokenizer as text encoder/classifier
        )
        print("Explainer LLM pipeline loaded successfully.")
    except Exception as e:
        print(f"Error loading explainer LLM: {e}")
        print("Falling back to dummy explanation generation (no LLM).")
        explainer_pipeline = None # Set to None if loading fails

    # Generate explanations for a few samples from the test set
    num_explanations_to_generate = min(20, len(test_dataset)) # Generate for up to 20 samples or less if dataset is smaller
    
    # Get a subset of the test dataset for explanation generation
    # It's better to pick samples that were actually classified as abusive by the model
    # For demonstration, we'll just take the first N samples.
    test_samples_for_explanation = [test_dataset[i] for i in range(num_explanations_to_generate)]

    generated_explanations_data = generate_explanations(
        model, explainer_pipeline, test_samples_for_explanation, DEVICE
    )
    
    print("\n--- Explanation Generation Complete ---")
    generated_df = pd.DataFrame(generated_explanations_data)
    print("Generated Explanations (first 5 rows):\n", generated_df.head(5))

    # Optional: Save generated explanations
    generated_df.to_csv(os.path.join(RESULTS_DIR, "generated_explanations.csv"), index=False)
    print(f"Generated explanations saved to {os.path.join(RESULTS_DIR, 'generated_explanations.csv')}")


if __name__ == "__main__":
    main()