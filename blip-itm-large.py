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
MODEL_NAME         = "Salesforce/blip-itm-large-coco"
BATCH_SIZE         = 32       
ACCUM_STEPS        = 4        
EPOCHS             = 8
# Differential Learning Rates: Low for vision (pretrained), higher for text/proj
LR_VISION          = 1e-6     
LR_TEXT            = 1e-5     
WEIGHT_DECAY       = 0.05     # Reduced from 0.5 to prevent underfitting
WARMUP_STEPS_RATIO = 0.1
PATIENCE           = 3
TEMPERATURE        = 0.07     
MAX_TEXT_LENGTH    = 128      
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR           = "./blip_itm_large_contrastive_fixed"

os.makedirs(SAVE_DIR, exist_ok=True)

dataset_handle = "simhadrisadaram/mimic-cxr-dataset/versions/2"
BASE_PATH  = kagglehub.dataset_download(dataset_handle)
TRAIN_CSV  = os.path.join(BASE_PATH, "mimic_cxr_aug_train.csv")
TEST_CSV   = os.path.join(BASE_PATH, "mimic_cxr_aug_validate.csv")
IMAGE_ROOT = os.path.join(BASE_PATH, "official_data_iccv_final")

# ======================================
# TRANSFORMS & DATA
# ======================================
_MEAN = [0.48145466, 0.4578275,  0.40821073]
_STD  = [0.26862954, 0.26130258, 0.27577711]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1), # Reduced jitter for medical
    transforms.ToTensor(),
    transforms.Normalize(mean=_MEAN, std=_STD),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_MEAN, std=_STD),
])

def clean_report(text: str) -> str:
    text = str(text).lower()
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            text = " ".join(str(s) for s in parsed)
    except Exception:
        pass
    if "findings:" in text:
        text = text.split("findings:")[1]
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
                path = os.path.join(IMAGE_ROOT, str(img_p).strip().replace("'", "").replace('"', ""))
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
    images, texts = zip(*batch)
    return torch.stack(images), list(texts)

# ======================================
# LOSS
# ======================================
def standard_infonce_loss(image_features, text_features, temperature=TEMPERATURE):
    """
    Standard symmetric cross-entropy. Much more stable for initializing 
    the contrastive space than aggressive hard-negative mining.
    """
    logit_scale = 1.0 / temperature
    logits = logit_scale * image_features @ text_features.T
    labels = torch.arange(len(image_features), device=logits.device)
    
    loss_i = nn.functional.cross_entropy(logits, labels)
    loss_t = nn.functional.cross_entropy(logits.T, labels)
    return (loss_i + loss_t) / 2

# ======================================
# TRAINING
# ======================================
def train():
    print(f"\nLoading {MODEL_NAME} …")
    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    model     = BlipForImageTextRetrieval.from_pretrained(MODEL_NAME).to(DEVICE)

    model.vision_model.encoder.gradient_checkpointing = True

    # Differential Optimization: Unfreeze vision tower but train it slowly
    for param in model.vision_model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW([
        {"params": model.vision_model.parameters(), "lr": LR_VISION},
        {"params": model.text_encoder.parameters(), "lr": LR_TEXT},
        {"params": model.vision_proj.parameters(), "lr": LR_TEXT},
        {"params": model.text_proj.parameters(), "lr": LR_TEXT},
    ], weight_decay=WEIGHT_DECAY)

    df_train_full = pd.read_csv(TRAIN_CSV)
    group_col = df_train_full["subject_id"] if "subject_id" in df_train_full.columns else df_train_full.index
    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    train_idx, val_idx = next(gss.split(df_train_full, groups=group_col))

    train_loader = DataLoader(
        CXRDataset(df_train_full.iloc[train_idx], train_transform),
        batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=8, pin_memory=True
    )
    val_loader = DataLoader(
        CXRDataset(df_train_full.iloc[val_idx], val_transform),
        batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=8, pin_memory=True
    )

    num_training_steps = len(train_loader) * EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(WARMUP_STEPS_RATIO * num_training_steps),
        num_training_steps=num_training_steps
    )
    scaler = GradScaler()

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")

        for step, (images, texts) in enumerate(pbar):
            images = images.to(DEVICE)
            text_inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_TEXT_LENGTH).to(DEVICE)

            with autocast(device_type="cuda"):
                image_features = model.vision_proj(model.vision_model(pixel_values=images).pooler_output)
                text_features = model.text_proj(model.text_encoder(input_ids=text_inputs["input_ids"], attention_mask=text_inputs["attention_mask"]).last_hidden_state[:, 0, :])

                image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-6)
                text_features  = text_features  / (text_features.norm(dim=-1, keepdim=True)  + 1e-6)

                loss = standard_infonce_loss(image_features, text_features)

            scaler.scale(loss / ACCUM_STEPS).backward()
            if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, texts in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                images = images.to(DEVICE)
                text_inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_TEXT_LENGTH).to(DEVICE)

                img_f = model.vision_proj(model.vision_model(pixel_values=images).pooler_output)
                txt_f = model.text_proj(model.text_encoder(input_ids=text_inputs["input_ids"], attention_mask=text_inputs["attention_mask"]).last_hidden_state[:, 0, :])

                img_f = img_f / (img_f.norm(dim=-1, keepdim=True) + 1e-6)
                txt_f = txt_f / (txt_f.norm(dim=-1, keepdim=True) + 1e-6)

                val_loss += standard_infonce_loss(img_f, txt_f).item()

        avg_val = val_loss / len(val_loader)
        print(f"\nEpoch {epoch+1} | Val Loss: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pt"))
            print("  ✓ Checkpoint saved.")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("  Early stopping triggered.")
                break

# ======================================
# EVALUATION (retrieval METEOR)
# ======================================
def evaluate():
    print("\nStarting Evaluation …")
    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    model     = BlipForImageTextRetrieval.from_pretrained(MODEL_NAME)
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "best_model.pt"), map_location="cpu", weights_only=True))
    model = model.to(DEVICE).eval()

    test_df = pd.read_csv(TEST_CSV)
    loader  = DataLoader(
        CXRDataset(test_df, val_transform),
        batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=8, pin_memory=True
    )

    img_embeds, txt_embeds, all_texts = [], [], []

    with torch.no_grad():
        for images, reports in tqdm(loader, desc="Encoding test set"):
            images = images.to(DEVICE)
            tokens = processor(text=reports, return_tensors="pt", padding=True, truncation=True, max_length=MAX_TEXT_LENGTH).to(DEVICE)

            i_f = model.vision_proj(model.vision_model(pixel_values=images).pooler_output)
            t_f = model.text_proj(model.text_encoder(input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"]).last_hidden_state[:, 0, :])
            
            img_embeds.append(i_f.cpu())
            txt_embeds.append(t_f.cpu())
            all_texts.extend(reports)

    img_embeds = torch.cat(img_embeds)
    txt_embeds = torch.cat(txt_embeds)

    img_embeds /= img_embeds.norm(dim=1, keepdim=True)
    txt_embeds /= txt_embeds.norm(dim=1, keepdim=True)

    similarity = img_embeds @ txt_embeds.T
    meteor_scores = []

    for i in tqdm(range(similarity.shape[0]), desc="Calculating METEOR"):
        top_idx = similarity[i].argmax().item()
        ref     = all_texts[i].split()
        pred    = all_texts[top_idx].split()
        meteor_scores.append(meteor_score([ref], pred))

    avg_meteor = sum(meteor_scores) / len(meteor_scores)
    print(f"\nFinal METEOR Score (retrieval): {avg_meteor:.4f}")
    return avg_meteor

if __name__ == "__main__":
    train()
    evaluate()
