import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import ast
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoProcessor, AutoModel
from nltk.translate.meteor_score import meteor_score
import kagglehub

# =====================================================
# CONFIG
# =====================================================

dataset_handle = "simhadrisadaram/mimic-cxr-dataset/versions/2"
BASE_PATH = kagglehub.dataset_download(dataset_handle)

# Construct your specific file paths dynamically using the returned BASE_PATH
CSV_FILE = os.path.join(BASE_PATH, "mimic_cxr_aug_train.csv")
IMAGE_ROOT = os.path.join(BASE_PATH, "official_data_iccv_final")

# Switched to SigLIP model
MODEL_NAME = "google/siglip2-giant-opt-patch16-384"

BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(42)

# =====================================================
# DATASET
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
        print(f"Total samples loaded: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, report = self.samples[idx]

        try:
            image = Image.open(image_path).convert("RGB")
        except:
            image = Image.new("RGB", (384, 384), color=(0, 0, 0)) # Updated fallback size for MedSigLIP

        return image, report


def collate_fn(batch):
    images, texts = zip(*batch)
    return list(images), list(texts)


# =====================================================
# EVALUATION
# =====================================================

def main():
    print(f"Loading {MODEL_NAME}...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    dataset = CXRDataset(CSV_FILE)

    # Keeping the same split to ensure we evaluate on the exact same test set
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    _, _, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size]
    )

    print(f"Test size: {len(test_dataset)}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    # -------------------------------------------------
    # Extract embeddings
    # -------------------------------------------------
    print("Extracting embeddings...")

    all_image_embeds = []
    all_text_embeds = []
    all_reports = []

    with torch.no_grad():
        for images, texts in tqdm(test_loader, desc="Embedding Extraction"):

            inputs = processor(
                text=texts,
                images=images,
                return_tensors="pt",
                padding="max_length", # SigLIP often prefers max_length padding
                truncation=True
            )

            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            outputs = model(**inputs)

            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds

            # Normalize embeddings
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

            all_image_embeds.append(image_embeds.cpu())
            all_text_embeds.append(text_embeds.cpu())
            all_reports.extend(texts)

    all_image_embeds = torch.cat(all_image_embeds)
    all_text_embeds = torch.cat(all_text_embeds)

    # -------------------------------------------------
    # Compute similarity matrix
    # -------------------------------------------------
    print("Computing similarity matrix...")
    similarity = all_image_embeds @ all_text_embeds.T

    # -------------------------------------------------
    # Compute METEOR
    # -------------------------------------------------
    print("Computing METEOR scores...")

    meteor_scores = []

    for i in tqdm(range(similarity.size(0)), desc="METEOR Evaluation"):

        top1 = similarity[i].argmax().item()

        reference = all_reports[i]
        candidate = all_reports[top1]

        # Tokenize for METEOR
        reference_tokens = reference.split()
        candidate_tokens = candidate.split()

        score = meteor_score([reference_tokens], candidate_tokens)
        meteor_scores.append(score)

        # Running average
        if (i + 1) % 1000 == 0:
            avg_meteor = sum(meteor_scores) / len(meteor_scores)
            tqdm.write(f"Processed {i+1} samples | Running Avg METEOR: {avg_meteor:.4f}")

    # -------------------------------------------------
    # Final Results
    # -------------------------------------------------
    avg_meteor = sum(meteor_scores) / len(meteor_scores)

    print("\n===== FINAL METEOR RESULTS =====")
    print(f"METEOR Score: {avg_meteor:.4f}")


if __name__ == "__main__":
    main()