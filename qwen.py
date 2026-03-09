import os
import ast
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, random_split, Subset
from nltk.translate.meteor_score import meteor_score
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import kagglehub
import random

# =====================================================
# CONFIGURATION
# =====================================================
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

dataset_handle = "simhadrisadaram/mimic-cxr-dataset/versions/2"
BASE_PATH = kagglehub.dataset_download(dataset_handle)
CSV_FILE = os.path.join(BASE_PATH, "mimic_cxr_aug_train.csv")
IMAGE_ROOT = os.path.join(BASE_PATH, "official_data_iccv_final")

# =====================================================
# DATASET (Replicated exactly to maintain split)
# =====================================================
class CXRDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        samples = []

        for _, row in df.iterrows():
            report = str(row["text"]).strip()
            if report == "" or report.lower() == "nan":
                report = "No findings."

            raw_paths = row["image"]

            try:
                image_list = ast.literal_eval(raw_paths)
                if not isinstance(image_list, list):
                    image_list = [image_list]
            except:
                image_list = [raw_paths]

            for img_path in image_list:
                img_path = str(img_path).strip().replace('"', '').replace("'", "")
                full_path = os.path.join(IMAGE_ROOT, img_path)

                if os.path.exists(full_path):
                    samples.append((full_path, report))

        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# =====================================================
# INFERENCE & EVALUATION
# =====================================================
def main():
    print("Loading dataset and splitting...")
    dataset = CXRDataset(CSV_FILE)
    
    # EXACT SAME SPLIT AS ORIGINAL SCRIPT
    torch.manual_seed(42)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    _, _, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size]
    )
    print(f"Test size: {len(test_dataset)}")

    random.seed(42) # Keep it reproducible
    subset_indices = random.sample(range(len(test_dataset)), 500)
    test_dataset = Subset(test_dataset, subset_indices)

    print(f"Loading {MODEL_NAME}...")
    # Load model in bfloat16 to save memory
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model.eval()

    meteor_scores = []
    
    print("Starting Qwen3-VL Zero-Shot Evaluation...")
    
    for i in tqdm(range(len(test_dataset)), desc="Evaluating"):
        image_path, reference_report = test_dataset[i]

        # 1. IDENTIFY THE CULPRIT: Print the path if it gets stuck
        if i >= 98: # Start printing right before where it usually hangs
            print(f"\n[Debug] Processing Index {i}: {image_path}")
            
        # Safely load image
        try:
            image = Image.open(image_path).convert("RGB")
            
            # 2. CAP THE RESOLUTION: Prevent massive attention matrix hangs
            image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            
        except Exception as e:
            print(f"\n[Warning] Failed to load {image_path}. Skipping. Error: {e}")
            continue # Skip this image entirely and move to the next one
        
        # Format the prompt exactly how Qwen expects it
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": "What are the radiological findings in this chest X-ray? Provide a short, clinical report without any introductory filler."}
                ]
            }
        ]
        
        # Prepare inputs using Qwen's vision processors
        text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(DEVICE)
        
        # Generate the report
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=50)
            
        # Extract only the newly generated tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        candidate_report = processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        # Compute METEOR
        reference_tokens = reference_report.split()
        candidate_tokens = candidate_report.split()
        
        score = meteor_score([reference_tokens], candidate_tokens)
        meteor_scores.append(score)

        # 3. MEMORY MANAGEMENT: Force PyTorch to clean up the garbage
        del inputs
        del generated_ids
        del generated_ids_trimmed
        torch.cuda.empty_cache()
        
        if (i + 1) % 500 == 0:
            avg_meteor = sum(meteor_scores) / len(meteor_scores)
            tqdm.write(f"Processed {i+1} samples | Running Avg METEOR: {avg_meteor:.4f}")

    # Final Results
    final_avg_meteor = sum(meteor_scores) / len(meteor_scores)
    print("\n===== FINAL GENERATIVE METEOR RESULTS =====")
    print(f"Zero-Shot Qwen3-VL METEOR Score: {final_avg_meteor:.4f}")

if __name__ == "__main__":
    main()