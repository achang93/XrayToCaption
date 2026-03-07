import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import ast
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import CLIPProcessor, CLIPModel, get_cosine_schedule_with_warmup
from torch.amp import autocast, GradScaler
from nltk.translate.meteor_score import meteor_score

# =====================================================
# CONFIG
# =====================================================

BASE_PATH = "/home/achang93/.cache/kagglehub/datasets/simhadrisadaram/mimic-cxr-dataset/versions/2"
CSV_FILE = os.path.join(BASE_PATH, "mimic_cxr_aug_train.csv")
IMAGE_ROOT = os.path.join(BASE_PATH, "official_data_iccv_final")

MODEL_NAME = "openai/clip-vit-large-patch14"

BATCH_SIZE = 32
GRAD_ACCUM_STEPS = 2
EPOCHS = 10

LR = 5e-6                 # smaller LR
WEIGHT_DECAY = 0.05       # stronger regularization
PATIENCE = 2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "./best_clip_large_model"

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
            image = Image.new("RGB", (224, 224), color=(0, 0, 0))

        return image, report


def collate_fn(batch):
    images, texts = zip(*batch)
    return list(images), list(texts)


# =====================================================
# TRAINING
# =====================================================

def main():

    print("Loading CLIP-Large...")
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)

    # Full fine-tuning
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    dataset = CXRDataset(CSV_FILE)

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(0.1 * total_steps)   # 10% warmup

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    scaler = GradScaler("cuda")

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(EPOCHS):

        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        # ================= TRAIN =================
        model.train()
        train_loss = 0
        train_bar = tqdm(train_loader, desc="Train")

        optimizer.zero_grad()

        for step, (images, texts) in enumerate(train_bar):

            inputs = processor(
                text=texts,
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(DEVICE)

            with autocast("cuda"):
                outputs = model(**inputs, return_loss=True)
                loss = outputs.loss / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:

                # Gradient clipping (important)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

                # Stabilize temperature
                model.logit_scale.data = torch.clamp(
                    model.logit_scale.data, max=4.6052
                )

            train_loss += loss.item() * GRAD_ACCUM_STEPS
            train_bar.set_postfix(loss=loss.item() * GRAD_ACCUM_STEPS)

        avg_train_loss = train_loss / len(train_loader)

        # ================= VALIDATION =================
        model.eval()
        val_loss = 0

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc="Validation")

            for images, texts in val_bar:

                inputs = processor(
                    text=texts,
                    images=images,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(DEVICE)

                with autocast("cuda"):
                    outputs = model(**inputs, return_loss=True)
                    loss = outputs.loss

                val_loss += loss.item()
                val_bar.set_postfix(loss=loss.item())

        avg_val_loss = val_loss / len(val_loader)

        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss:   {avg_val_loss:.4f}")

        # ================= EARLY STOPPING =================
        if avg_val_loss < best_val_loss:
            print("Validation improved. Saving best model...")
            best_val_loss = avg_val_loss
            patience_counter = 0

            os.makedirs(SAVE_DIR, exist_ok=True)
            model.save_pretrained(SAVE_DIR)
            processor.save_pretrained(SAVE_DIR)

        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    print("\nTraining complete.")
    print(f"Best validation loss: {best_val_loss:.4f}")

    ####################################################
    #####################################################
    # EVALUATION BEGINS HERE
    ####################################################
    ####################################################
    print("Loading fine-tuned model...")
    model = CLIPModel.from_pretrained(SAVE_DIR).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(SAVE_DIR)
    model.eval()


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
                padding=True,
                truncation=True
            )

            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            outputs = model(**inputs)

            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds

            # Normalize
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

