import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
from transformers import CLIPProcessor, AutoTokenizer
from sentence_transformers import SentenceTransformer # For the TEXT_EMBEDDER_MODEL_NAME tokenizer if it's SBERT

# Define image transformations (CLIP's default expects 224x224, normalization)
# Ensure this matches the CLIPProcessor's internal transforms
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet normalization
])

class MMCAD_Dataset(Dataset):
    def __init__(self, csv_file, img_dir, clip_processor, text_tokenizer, max_seq_length):
        """
        Args:
            csv_file (string): Path to the CSV file with meme annotations (e.g., DACAM structure).
            img_dir (string): Directory with all the meme images.
            clip_processor (CLIPProcessor): Preprocessor for CLIP visual encoder.
            text_tokenizer (AutoTokenizer or SentenceTransformer): Tokenizer for the text encoder.
            max_seq_length (int): Maximum sequence length for text tokenization.
        """
        self.df = pd.read_csv(csv_file).dropna(subset=["title", "filename", "label"])
        self.img_dir = img_dir
        self.clip_processor = clip_processor
        self.text_tokenizer = text_tokenizer
        self.max_seq_length = max_seq_length

        # DACAM specific: Assuming 'title' is the extracted OCR text + title
        # And 'label' is the binary classification label (0/1)
        # You would adapt this based on the actual DACAM CSV schema.

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.img_dir, self.df.iloc[idx]['filename'])
        try:
            image = Image.open(img_name).convert("RGB")
            # Apply common transforms outside CLIPProcessor for direct tensor input,
            # or rely fully on CLIPProcessor's internal logic.
            # Using clip_processor.preprocess handles resizing and normalization correctly.
        except FileNotFoundError:
            # Handle missing images: e.g., skip, return dummy, or log error
            print(f"Warning: Image not found for {img_name}. Skipping sample.")
            return self.__getitem__((idx + 1) % len(self)) # Load next item

        text_content = str(self.df.iloc[idx]['title']) # Combined OCR text + title

        label = int(self.df.iloc[idx]['label']) # 0: non-abusive, 1: abusive

        # Process image for CLIP
        # clip_processor expects PIL Image and handles all preprocessing internally
        image_inputs = self.clip_processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0) # Remove batch dim

        # Process text for LLM-based encoder
        # Check if tokenizer is from SentenceTransformer or AutoTokenizer
        if isinstance(self.text_tokenizer, SentenceTransformer):
            # SentenceTransformer.encode returns embeddings directly. For raw token IDs, use its underlying tokenizer.
            # For this pipeline, we need token IDs for the AutoModel, not embeddings yet.
            raise ValueError("SentenceTransformer object passed as text_tokenizer. Please pass its underlying tokenizer or AutoTokenizer.")
        
        text_inputs = self.text_tokenizer(
            text_content,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_length
        )
        
        # Squeeze batch dimension for single item return from tokenizer
        text_input_ids = text_inputs['input_ids'].squeeze(0)
        text_attention_mask = text_inputs['attention_mask'].squeeze(0)
        
        sample = {
            'pixel_values': image_inputs,
            'input_ids': text_input_ids,
            'attention_mask': text_attention_mask,
            'labels': torch.tensor(label, dtype=torch.long),
            'raw_text_content': text_content # Keep raw text for explanation generation
        }

        return sample

# --- Example Usage (Conceptual) ---
if __name__ == '__main__':
    import config
    
    # Initialize CLIP processor
    clip_processor_instance = CLIPProcessor.from_pretrained(config.CLIP_MODEL_NAME)
    
    # Initialize tokenizer for the text encoder (e.g., LLaMA 2's tokenizer)
    # Using AutoTokenizer which is typical for HuggingFace models
    tokenizer_instance = AutoTokenizer.from_pretrained(config.LLM_CLASSIFIER_MODEL_NAME)
    if tokenizer_instance.pad_token is None:
        tokenizer_instance.pad_token = tokenizer_instance.eos_token
        tokenizer_instance.padding_side = "right"

    # --- DACAM Data Specifics Placeholder ---
    # In a real DACAM setup, you'd ensure config.CSV_FILE points to your
    # primary annotation file and config.IMG_DIR to the image folder.
    # For a quick dummy test, create dummy CSV and images if not available.
    if not os.path.exists(config.CSV_FILE):
        print(f"!!! WARNING: {config.CSV_FILE} not found. Creating a dummy CSV for testing. !!!")
        dummy_df = pd.DataFrame({
            'filename': [f"dummy_img_{i}.jpg" for i in range(10)],
            'title': [f"This is dummy meme text {i}" for i in range(10)],
            'label': [i % 2 for i in range(10)] # Alternating labels
        })
        dummy_df.to_csv(config.CSV_FILE, index=False)
        
        os.makedirs(config.IMG_DIR, exist_ok=True)
        for i in range(10):
            dummy_img_path = os.path.join(config.IMG_DIR, f"dummy_img_{i}.jpg")
            if not os.path.exists(dummy_img_path):
                Image.new('RGB', (224, 224), color = 'red').save(dummy_img_path)
        print("Dummy CSV and images created.")

    # Create dataset instance
    dataset = MMCAD_Dataset(
        csv_file=config.CSV_FILE,
        img_dir=config.IMG_DIR,
        clip_processor=clip_processor_instance,
        text_tokenizer=tokenizer_instance,
        max_seq_length=config.MAX_SEQ_LENGTH
    )

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    print(f"Dataset size: {len(dataset)}")
    for i, batch in enumerate(dataloader):
        print(f"Batch {i+1} structure:")
        print("Pixel Values (images):", batch['pixel_values'].shape)
        print("Input IDs (text):", batch['input_ids'].shape)
        print("Attention Mask (text):", batch['attention_mask'].shape)
        print("Labels:", batch['labels'].shape)
        print("Raw Text Content (first item):", batch['raw_text_content'][0])
        if i == 0: # Print only first batch details
            break