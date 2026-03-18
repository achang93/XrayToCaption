"""
BLIP Contrastive Pipeline
=========================
Adapted from BiomedCLIP contrastive pipeline.
Uses Salesforce/blip-itm-base-coco as a dual encoder via BlipForImageTextRetrieval:
  - model.get_image_features()  →  image embeddings  (tensor, no .pooler_output needed)
  - model.get_text_features()   →  text embeddings   (tensor, no .pooler_output needed)
Trained with symmetric InfoNCE (CLIP-style) loss.
Evaluated with retrieval-based METEOR: for each image, retrieve the
top-1 most similar report by cosine similarity and compute METEOR
against the ground-truth report.

Key differences from BiomedCLIP version:
  - BlipForImageTextRetrieval used (not BlipModel, which has randomly-initialised
    text_model weights when loaded from blip-image-captioning-base).
    blip-itm-base-coco is pretrained end-to-end for contrastive retrieval.
  - open_clip replaced with transformers BlipForImageTextRetrieval + BlipProcessor
  - model.visual  →  model.vision_model  (freeze target)
  - logit_scale is fixed (no learned temperature in BlipForImageTextRetrieval)
  - Tokenization done in training loop via processor, not collate_fn
  - BATCH_SIZE reduced to 64 (BLIP is heavier than BiomedCLIP ViT-B/16)
"""

import os
import ast
import re
import torch
import torch.nn as nn
import pandas as pd
import kagglehub
import nltk
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from torch.amp import autocast, GradScaler
from transformers import BlipProcessor, BlipForImageTextRetrieval, get_cosine_schedule_with_warmup
from nltk.translate.meteor_score import meteor_score

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ======================================
# CONFIGURATION
# ======================================
MODEL_NAME         = "Salesforce/blip-itm-base-coco"   # ViT-B, fits on 24GB at batch 256
BATCH_SIZE         = 128      # 256 OOMs on 24GB with base model; 128 is the safe ceiling
EPOCHS             = 8
LR                 = 5e-6     # Lowered — 3e-5 was overfitting from epoch 1
WEIGHT_DECAY       = 0.5      # Increased — val loss was climbing every epoch
WARMUP_STEPS_RATIO = 0.1
PATIENCE           = 3
TEMPERATURE        = 0.07     # Fixed temperature (BLIP has no learned logit_scale)
FREEZE_VISION      = True     # Keep vision tower frozen permanently
# Unfreezing was causing val loss to worsen — vision tower overfits to training images.
# Only the text encoder and projection layers train.
UNFREEZE_EPOCH     = 999      # Effectively never — set to a real epoch number to re-enable
LR_VISION_UNFROZEN = 1e-6     # Only used if UNFREEZE_EPOCH is reached
MAX_TEXT_LENGTH    = 128      # CXR reports are short; 256 wastes attention memory
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR           = "./blip_itm_base_contrastive"

os.makedirs(SAVE_DIR, exist_ok=True)

# Data paths
dataset_handle = "simhadrisadaram/mimic-cxr-dataset/versions/2"
BASE_PATH  = kagglehub.dataset_download(dataset_handle)
TRAIN_CSV  = os.path.join(BASE_PATH, "mimic_cxr_aug_train.csv")
TEST_CSV   = os.path.join(BASE_PATH, "mimic_cxr_aug_validate.csv")
IMAGE_ROOT = os.path.join(BASE_PATH, "official_data_iccv_final")

# ======================================
# TRANSFORMS
# ======================================
# BLIP was pretrained with these exact ImageNet-style stats.
# Using processor.feature_extractor stats avoids any mismatch.
_MEAN = [0.48145466, 0.4578275,  0.40821073]
_STD  = [0.26862954, 0.26130258, 0.27577711]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=_MEAN, std=_STD),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_MEAN, std=_STD),
])

# ======================================
# DATA PROCESSING
# ======================================
def clean_report(text: str) -> str:
    """
    Extract findings section only.

    Impression is deliberately excluded: it is a radiologist summary written
    to be concise and retrievable, so including it inflates METEOR by making
    retrieval trivially easy. Findings text requires the model to understand
    image content rather than match summary keywords.

    Both "findings:" and "impression:" headers are stripped so neither
    section label leaks into the text embeddings.
    """
    text = str(text).lower()
    # Unwrap MIMIC list-string format e.g. "['Findings: ...', 'Impression: ...']"
    try:
        import ast as _ast
        parsed = _ast.literal_eval(text)
        if isinstance(parsed, list):
            text = " ".join(str(s) for s in parsed)
    except Exception:
        pass
    # Extract findings section; fall back to full text if header absent
    if "findings:" in text:
        text = text.split("findings:")[1]
    # Strip impression section entirely — do not keep it
    if "impression:" in text:
        text = text.split("impression:")[0]
    text = re.sub(r"[^a-z0-9\s\.]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class CXRDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        self.transform = transform
        self.samples   = []
        for _, row in df.iterrows():
            report = clean_report(row["text"])
            if len(report) < 10:
                continue
            try:
                images = ast.literal_eval(row["image"])
                if not isinstance(images, list):
                    images = [images]
            except Exception:
                images = [row["image"]]

            for img_p in images:
                path = os.path.join(
                    IMAGE_ROOT,
                    str(img_p).strip().replace("'", "").replace('"', ""),
                )
                if os.path.exists(path):
                    self.samples.append((path, report))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, text = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224))
        return self.transform(image), text


def collate_fn(batch):
    """Stack images; keep texts as a list — tokenized in the training loop."""
    images, texts = zip(*batch)
    return torch.stack(images), list(texts)


# ======================================
# LOSS
# ======================================
def contrastive_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    temperature: float = TEMPERATURE,
) -> torch.Tensor:
    """
    Symmetric InfoNCE (CLIP-style) loss.
    image_features and text_features must already be L2-normalised.
    """
    logit_scale = 1.0 / temperature
    logits      = logit_scale * image_features @ text_features.T   # (B, B)
    labels      = torch.arange(len(image_features), device=logits.device)
    loss_i      = nn.functional.cross_entropy(logits,   labels)
    loss_t      = nn.functional.cross_entropy(logits.T, labels)
    return (loss_i + loss_t) / 2


# ======================================
# TRAINING
# ======================================
def train():
    print(f"\nLoading {MODEL_NAME} …")
    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    model     = BlipForImageTextRetrieval.from_pretrained(MODEL_NAME).to(DEVICE)

    # Freeze vision tower — keeps pretrained medical visual features intact
    if FREEZE_VISION:
        for param in model.vision_model.parameters():
            param.requires_grad = False
        total     = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Vision tower frozen.  Trainable: {trainable:,} / {total:,} params")

    # ── Data ──────────────────────────────────────────────────────────────────
    # Use the provided train CSV for train/val, with a patient-level val split.
    # TEST_CSV is the held-out validate CSV provided by the dataset authors.
    df_train_full = pd.read_csv(TRAIN_CSV)
    group_col = (
        df_train_full["subject_id"]
        if "subject_id" in df_train_full.columns
        else df_train_full.index
    )
    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    train_idx, val_idx = next(gss.split(df_train_full, groups=group_col))
    print(f"  Split — train={len(train_idx)} val={len(val_idx)}")

    train_loader = DataLoader(
        CXRDataset(df_train_full.iloc[train_idx], train_transform),
        batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=8,
        pin_memory=True, persistent_workers=True, prefetch_factor=2,
    )
    val_loader = DataLoader(
        CXRDataset(df_train_full.iloc[val_idx], val_transform),
        batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=8,
        pin_memory=True, persistent_workers=True, prefetch_factor=2,
    )

    # ── Optimiser & scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    num_training_steps = len(train_loader) * EPOCHS
    num_warmup_steps   = int(WARMUP_STEPS_RATIO * num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    scaler = GradScaler()

    best_val_loss    = float("inf")
    patience_counter = 0

    print(f"\nTraining for up to {EPOCHS} epochs (patience={PATIENCE}) …\n")

    for epoch in range(EPOCHS):
        # ── Unfreeze vision tower after warmup ────────────────────────────────
        if FREEZE_VISION and epoch == UNFREEZE_EPOCH:
            for param in model.vision_model.parameters():
                param.requires_grad = True
            # Add vision params to optimizer with a much lower LR
            optimizer.add_param_group({
                "params": model.vision_model.parameters(),
                "lr": LR_VISION_UNFROZEN,
            })
            print(f"  [Epoch {epoch+1}] Vision tower unfrozen (lr={LR_VISION_UNFROZEN:.0e})")

        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")

        for images, texts in pbar:
            images = images.to(DEVICE)

            # Tokenize text in the training loop — processor handles padding/truncation
            text_inputs = processor(
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_TEXT_LENGTH,
            ).to(DEVICE)

            with autocast(device_type="cuda"):
                # Call submodules directly — use_itm_head=False does not
                # populate image_embeds in this transformers version.
                image_features = model.vision_proj(
                    model.vision_model(pixel_values=images).pooler_output
                )
                text_features = model.text_proj(
                    model.text_encoder(
                        input_ids=text_inputs["input_ids"],
                        attention_mask=text_inputs["attention_mask"],
                    ).last_hidden_state[:, 0, :]  # CLS token
                )

                # L2 normalise with epsilon for numerical stability
                image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-6)
                text_features  = text_features  / (text_features.norm(dim=-1, keepdim=True)  + 1e-6)

                loss = contrastive_loss(image_features, text_features)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr":   f"{scheduler.get_last_lr()[0]:.2e}",
            })

        # ── Validation ─────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, texts in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                images      = images.to(DEVICE)
                text_inputs = processor(
                    text=texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=MAX_TEXT_LENGTH,
                ).to(DEVICE)

                img_f = model.vision_proj(
                    model.vision_model(pixel_values=images).pooler_output
                )
                txt_f = model.text_proj(
                    model.text_encoder(
                        input_ids=text_inputs["input_ids"],
                        attention_mask=text_inputs["attention_mask"],
                    ).last_hidden_state[:, 0, :]
                )

                img_f = img_f / (img_f.norm(dim=-1, keepdim=True) + 1e-6)
                txt_f = txt_f / (txt_f.norm(dim=-1, keepdim=True) + 1e-6)

                val_loss += contrastive_loss(img_f, txt_f).item()

        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)
        print(f"\nEpoch {epoch+1} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss    = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pt"))
            print("  ✓ Checkpoint saved.")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{PATIENCE})")
            if patience_counter >= PATIENCE:
                print("  Early stopping triggered.")
                break


# ======================================
# EVALUATION (retrieval METEOR)
# ======================================
def evaluate():
    """
    Retrieval-based METEOR evaluation — identical logic to BiomedCLIP version.
    For each image embedding, find the top-1 most similar text embedding by
    cosine similarity, then compute METEOR between retrieved and ground-truth report.

    Note: the same inflation caveat applies here as in generative METEOR.
    Because most reports are short and contain similar clinical language,
    retrieval scores can be inflated by accidentally retrieving a normal report
    for another normal image. Filter to fracture-positive cases for a harder eval.
    """
    print("\nStarting Evaluation …")
    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    model     = BlipForImageTextRetrieval.from_pretrained(MODEL_NAME)
    model.load_state_dict(
        torch.load(os.path.join(SAVE_DIR, "best_model.pt"), map_location="cpu")
    )
    model = model.to(DEVICE).eval()

    test_df = pd.read_csv(TEST_CSV)
    loader  = DataLoader(
        CXRDataset(test_df, val_transform),
        batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=8,
        pin_memory=True,
    )

    img_embeds, txt_embeds, all_texts = [], [], []

    with torch.no_grad():
        for images, reports in tqdm(loader, desc="Encoding test set"):
            images = images.to(DEVICE)
            tokens = processor(
                text=reports,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_TEXT_LENGTH,
            ).to(DEVICE)

            i_f = model.vision_proj(
                model.vision_model(pixel_values=images).pooler_output
            )
            t_f = model.text_proj(
                model.text_encoder(
                    input_ids=tokens["input_ids"],
                    attention_mask=tokens["attention_mask"],
                ).last_hidden_state[:, 0, :]
            )
            img_embeds.append(i_f.cpu())
            txt_embeds.append(t_f.cpu())
            all_texts.extend(reports)

    img_embeds = torch.cat(img_embeds)
    txt_embeds = torch.cat(txt_embeds)

    img_embeds /= img_embeds.norm(dim=1, keepdim=True)
    txt_embeds /= txt_embeds.norm(dim=1, keepdim=True)

    # Full similarity matrix — (N_images, N_texts)
    similarity    = img_embeds @ txt_embeds.T
    meteor_scores = []

    for i in tqdm(range(similarity.shape[0]), desc="Calculating METEOR"):
        top_idx = similarity[i].argmax().item()
        ref     = all_texts[i].split()
        pred    = all_texts[top_idx].split()
        meteor_scores.append(meteor_score([ref], pred))

    avg_meteor = sum(meteor_scores) / len(meteor_scores)
    print(f"\nFinal METEOR Score (retrieval): {avg_meteor:.4f}")
    print(
        f"  (N={len(meteor_scores)};  perfect retrieval would score ~1.0;"
        f"  random baseline ~{1/len(meteor_scores):.4f})"
    )
    return avg_meteor


if __name__ == "__main__":
    train()
    evaluate()
