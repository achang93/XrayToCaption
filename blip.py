"""
InstructBLIP Fracture Classifier + Localization Description Generator
=====================================================================
Architecture:
  - InstructBLIP: ViT-G vision encoder -> Q-Former -> Vicuna-7B (or FlanT5-XL)
  - We use Salesforce/instructblip-flan-t5-xl (smaller, no need for 7B GPU RAM)
  - LoRA on the language model (T5) decoder layers
  - Classification head on Q-Former output (mean-pooled query tokens)
  - Dual loss: FocalLoss (classification) + LM cross-entropy (generation)

InstructBLIP vs BLIP differences:
  - Q-Former extracts 32 visual query tokens conditioned on the text instruction
  - Processor is InstructBlipProcessor, not BlipProcessor
  - Model forward needs 'qformer_input_ids' in addition to pixel_values/input_ids
  - Generation prompt is a full instruction string, not a partial caption
  - Hidden dim comes from Q-Former query size (768), not vision encoder (1408)
"""

import os
import re
import ast
import math
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import kagglehub
import numpy as np
import nltk
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
import torchvision.transforms as T
from peft import LoraConfig, get_peft_model, TaskType

nltk.download("wordnet",   quiet=True)
nltk.download("omw-1.4",   quiet=True)
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_PATH  = kagglehub.dataset_download("simhadrisadaram/mimic-cxr-dataset")
TRAIN_CSV  = os.path.join(BASE_PATH, "mimic_cxr_aug_train.csv")
VAL_CSV    = os.path.join(BASE_PATH, "mimic_cxr_aug_validate.csv")
IMAGE_ROOT = os.path.join(BASE_PATH, "official_data_iccv_final")

# instructblip-flan-t5-xl needs ~14GB VRAM; use vicuna only if you have 40GB+
MODEL_NAME  = "Salesforce/instructblip-flan-t5-xl"

BATCH_SIZE         = 8     # 8 x 8 = effective batch 64; bfloat16 fits 24GB
ACCUMULATION_STEPS = 8     # effective batch = 64
EPOCHS             = 7
LR_CLS             = 3e-4
LR_LORA            = 1e-4  # T5 is larger; use smaller LR
LAMBDA_GEN         = 0.5
MAX_GEN_TOKENS     = 64
PATIENCE           = 3
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP            = torch.cuda.is_available()   # autocast (safe with bfloat16)
USE_SCALER         = False  # GradScaler is float16-only; bfloat16 does not need it
SAVE_DIR           = "./instructblip_fracture"
DEBUG_MODE         = False
IGNORE_INDEX       = -100
GEN_EVAL_EVERY     = 2
NUM_WORKERS        = 8

# InstructBLIP instruction — full natural language instruction fed to Q-Former
# This is different from BLIP which used a short prefix.
INSTRUCTION = (
    "You are a radiologist. "
    "Examine this chest X-ray and identify any fractures. "
    "If a fracture is present, describe its location precisely."
)
# Pre-tokenized instruction cache — populated in main() once processor is loaded.
# Avoids re-tokenizing the same instruction string on every batch in collate_fn.
_INSTR_IDS:      torch.Tensor | None = None
_INSTR_LEN:      int                 = 0

FRACTURE_KEYWORDS = ["fracture", "broken", "comminuted", "displaced"]

NEGATION_PATTERNS = [
    r'\bno\b', r'\bnot\b', r'\bwithout\b', r'\bdenies\b', r'\bfree\s+of\b',
    r'\bnegative\s+for\b', r'\babsence\s+of\b', r'\bno\s+evidence\s+of\b',
    r'\bnot\s+identified\b', r'\bnot\s+seen\b', r'\bnot\s+visualized\b',
    r'\bis\s+not\b', r'\bare\s+not\b', r'\bunchanged\b', r'\bnone\b',
    r'\bnever\b', r'\bno\s+acute\b', r'\bno\s+new\b',
]

LOCATION_KEYWORDS = [
    "rib", "clavicle", "scapula", "humerus", "femur", "tibia", "fibula",
    "radius", "ulna", "vertebra", "spine", "pelvis", "hip", "knee",
    "shoulder", "wrist", "ankle", "skull", "sternum", "left", "right",
    "bilateral", "proximal", "distal", "lateral", "medial", "anterior",
    "posterior", "upper", "lower", "mid",
]


# ─────────────────────────────────────────────
# TEXT UTILITIES
# ─────────────────────────────────────────────

def _unwrap_text(text: str) -> str:
    text = str(text).strip()
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return " ".join(str(s) for s in parsed if s)
    except Exception:
        pass
    return text


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r'[.;!\n]+|\b\d+\.\s+', text) if s.strip()]


def _get_impression_section(text: str) -> str:
    m = re.search(r'impression\s*[:\s]+(.*?)(?:findings|$)', text, re.DOTALL)
    return m.group(1).strip() if m else ""


def _sentence_has_affirmed_fracture(sentence: str) -> bool:
    s = sentence.lower()
    for kw in FRACTURE_KEYWORDS:
        for m in re.finditer(re.escape(kw), s):
            preceding = s[max(0, m.start() - 60): m.start()]
            if not any(re.search(pat, preceding) for pat in NEGATION_PATTERNS):
                return True
    return False


def is_fracture(text: str) -> bool:
    text = _unwrap_text(text).lower().strip()
    if not any(kw in text for kw in FRACTURE_KEYWORDS):
        return False
    return any(_sentence_has_affirmed_fracture(s) for s in _split_sentences(text) if s)


def extract_location_description(report_text: str) -> str:
    text = _unwrap_text(report_text).lower().strip()
    if not any(kw in text for kw in FRACTURE_KEYWORDS):
        return "No fracture identified."
    impression = _get_impression_section(text)
    if impression:
        imp_affirmed = [s for s in _split_sentences(impression) if _sentence_has_affirmed_fracture(s)]
        if imp_affirmed:
            with_loc = [s for s in imp_affirmed if any(loc in s for loc in LOCATION_KEYWORDS)]
            best = with_loc[0] if with_loc else imp_affirmed[0]
            best = re.sub(r'\s+', ' ', best).strip()
            return best[:120].rsplit(' ', 1)[0] if len(best) > 120 else best
    affirmed = [s for s in _split_sentences(text) if _sentence_has_affirmed_fracture(s)]
    if not affirmed:
        return "No fracture identified."
    with_loc = [s for s in affirmed if any(loc in s for loc in LOCATION_KEYWORDS)]
    best = with_loc[0] if with_loc else affirmed[0]
    best = re.sub(r'\s+', ' ', best).strip()
    return best[:120].rsplit(' ', 1)[0] if len(best) > 120 else best


def extract_all_fracture_sentences(report_text: str) -> list[str]:
    text = _unwrap_text(report_text).lower().strip()
    if not any(kw in text for kw in FRACTURE_KEYWORDS):
        return ["No fracture identified."]
    affirmed = [re.sub(r'\s+', ' ', s).strip()
                for s in _split_sentences(text) if _sentence_has_affirmed_fracture(s)]
    seen, unique = set(), []
    for s in affirmed:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique if unique else ["No fracture identified."]


# ─────────────────────────────────────────────
# LOSS & AUGMENTATION
# ─────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce  = F.cross_entropy(inputs, targets, reduction='none')
        pt  = torch.exp(-ce)
        return (self.alpha * (1 - pt) ** self.gamma * ce).mean()


class ApplyCLAHE:
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)):
        self.clip_limit     = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img: Image.Image) -> Image.Image:
        arr   = np.array(img.convert('L'))
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        return Image.fromarray(clahe.apply(arr)).convert('RGB')


TRAIN_AUG = T.Compose([
    ApplyCLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),
    T.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),  # InstructBLIP uses 224
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=10),
])


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────

def extract_patient_id(img_field: str) -> str:
    try:
        path = ast.literal_eval(img_field)[0] if str(img_field).startswith("[") else img_field
    except Exception:
        path = img_field
    parts      = str(path).replace("\\", "/").split("/")
    candidates = [p for p in parts if re.match(r'^p\d{5,}$', p)]
    return candidates[0] if candidates else parts[0]


def get_patient_level_splits(train_csv: str, val_csv: str):
    df_a   = pd.read_csv(train_csv)
    df_b   = pd.read_csv(val_csv)
    all_df = pd.concat([df_a, df_b]).drop_duplicates().reset_index(drop=True)
    all_df["text"]       = all_df["text"].astype(str).fillna("")
    all_df["patient_id"] = all_df["image"].apply(extract_patient_id)
    patients            = all_df["patient_id"].unique()
    train_pts, temp_pts = train_test_split(patients, test_size=0.25, random_state=42)
    val_pts,   test_pts = train_test_split(temp_pts, test_size=0.60, random_state=42)
    train_df = all_df[all_df["patient_id"].isin(train_pts)].reset_index(drop=True)
    val_df   = all_df[all_df["patient_id"].isin(val_pts)].reset_index(drop=True)
    test_df  = all_df[all_df["patient_id"].isin(test_pts)].reset_index(drop=True)
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        n = df["text"].apply(is_fracture).sum()
        print(f"    {name:<6} — {len(df):5d} samples  |  fracture={n} ({100*n/len(df):.1f}%)")
    return train_df, val_df, test_df


def compute_sample_weights(df: pd.DataFrame) -> torch.Tensor:
    labels       = df["text"].apply(lambda t: 1 if is_fracture(t) else 0).values
    class_counts = np.bincount(labels)
    weights      = torch.tensor([1.0 / class_counts[l] for l in labels], dtype=torch.float)
    print(f"    Weights — normal={1/class_counts[0]:.5f}  fracture={1/class_counts[1]:.5f}")
    return weights


class FractureDataset(Dataset):
    def __init__(self, df: pd.DataFrame, processor: InstructBlipProcessor, augment: bool = False):
        self.df        = df.reset_index(drop=True)
        self.processor = processor
        self.augment   = augment

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            img_list = ast.literal_eval(row["image"])
            img_path = img_list[0] if isinstance(img_list, list) else img_list
        except Exception:
            img_path = row["image"]
        full_path = os.path.join(IMAGE_ROOT, str(img_path).strip().replace("'", "").replace('"', ""))
        try:
            image = Image.open(full_path).convert("RGB")
            if self.augment:
                image = TRAIN_AUG(image)
        except Exception:
            image = Image.new("RGB", (224, 224), color=(0, 0, 0))

        cls_label = 1 if is_fracture(str(row["text"])) else 0
        loc_desc  = extract_location_description(str(row["text"]))
        all_refs  = extract_all_fracture_sentences(str(row["text"]))

        return image, cls_label, loc_desc, all_refs  # return PIL image, process in collate


def collate_fn(batch, processor: InstructBlipProcessor):
    """
    InstructBLIP collate.
    Optimisations vs naive version:
      - Instruction is pre-tokenized once (_INSTR_IDS/_INSTR_LEN) — no
        repeated tokenizer calls for the fixed instruction string.
      - Only TWO processor calls: one for image+instruction (Q-Former),
        one for full text (LM labels). Labels masked using cached instr length.
    """
    images, cls_labels, loc_descs, all_refs_batch = zip(*batch)
    cls_labels = torch.tensor(cls_labels, dtype=torch.long)
    B          = len(images)

    # ── Image + instruction encoding (Q-Former input) ─────────────────
    enc = processor(
        images         = list(images),
        text           = [INSTRUCTION] * B,
        return_tensors = "pt",
        padding        = True,
        truncation     = True,
    )

    # ── Full sequence encoding for LM teacher forcing ─────────────────
    full_texts = [f"{INSTRUCTION} {d}" for d in loc_descs]
    lm_enc = processor.tokenizer(
        full_texts,
        return_tensors = "pt",
        padding        = True,
        truncation     = True,
        max_length     = MAX_GEN_TOKENS + 64,
    )

    # ── Build labels using cached instruction length ───────────────────
    # _INSTR_LEN is set once in main() — avoids re-tokenizing every batch.
    labels = lm_enc["input_ids"].clone()
    labels[:, :_INSTR_LEN] = IGNORE_INDEX          # mask instruction tokens
    labels[lm_enc["input_ids"] == processor.tokenizer.pad_token_id] = IGNORE_INDEX

    return (
        enc["pixel_values"],
        enc["input_ids"],
        enc["attention_mask"],
        enc.get("qformer_input_ids"),
        enc.get("qformer_attention_mask"),
        cls_labels,
        lm_enc["input_ids"],
        lm_enc["attention_mask"],
        labels,
        list(loc_descs),
        list(all_refs_batch),
    )


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────

class InstructBLIPDualHead(nn.Module):
    """
    InstructBLIP with a classification head attached to Q-Former output.

    The Q-Former produces `num_query_tokens` (32) query vectors of size
    `qformer_hidden_size` (768). We mean-pool these and pass through an MLP
    for binary fracture classification. The full InstructBLIP model handles
    the generation path unchanged.
    """
    def __init__(self, model_name: str, dropout: float = 0.3):
        super().__init__()

        # Load base model — use float32 for stability, AMP handles mixed precision
        self.base = InstructBlipForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype = torch.float32,
        )

        # Apply LoRA to the language model (T5 encoder-decoder or Vicuna)
        # target_modules differ by backbone: T5 uses q/v, Vicuna uses q_proj/v_proj
        lm = self.base.language_model
        is_t5 = hasattr(lm, 'encoder')  # T5 has encoder/decoder, Vicuna is decoder-only
        target_mods = ["q", "v"] if is_t5 else ["q_proj", "v_proj"]
        task_type   = TaskType.SEQ_2_SEQ_LM if is_t5 else TaskType.CAUSAL_LM

        lora_cfg = LoraConfig(
            r             = 16,
            lora_alpha    = 32,
            target_modules = target_mods,
            lora_dropout  = 0.1,
            bias          = "none",
            task_type     = task_type,
        )
        self.base.language_model = get_peft_model(lm, lora_cfg)
        self.base.language_model.print_trainable_parameters()

        # Freeze vision encoder and Q-Former — only LoRA + classifier train
        for param in self.base.vision_model.parameters():
            param.requires_grad = False
        for param in self.base.qformer.parameters():
            param.requires_grad = False

        # Gradient checkpointing — recomputes activations during backward
        # instead of storing them. Saves ~30-40% activation memory.
        if hasattr(self.base.vision_model, 'gradient_checkpointing_enable'):
            self.base.vision_model.gradient_checkpointing_enable()
        if hasattr(self.base.language_model, 'gradient_checkpointing_enable'):
            self.base.language_model.gradient_checkpointing_enable()

        # Classification head on Q-Former output (768-dim query vectors)
        qf_hidden = self.base.qformer.config.hidden_size  # 768
        self.classifier = nn.Sequential(
            nn.LayerNorm(qf_hidden),
            nn.Linear(qf_hidden, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128),       nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 2),
        )
        for p in self.classifier.parameters():
            p.requires_grad = True
        self.classifier = self.classifier.to(torch.bfloat16)

    def forward(self,
                pixel_values,
                input_ids,            # instruction ids (for Q-Former conditioning)
                attention_mask,
                qformer_input_ids,
                qformer_attention_mask,
                lm_input_ids,         # full sequence ids (for LM forward)
                lm_attention_mask,
                labels=None):

        # ── Q-Former forward for classification ───────────────────────
        # Get image features from vision encoder
        vision_outputs = self.base.vision_model(pixel_values=pixel_values)
        image_embeds   = vision_outputs.last_hidden_state           # (B, 257, 1408)
        image_attn     = torch.ones(image_embeds.shape[:2],
                                    device=image_embeds.device, dtype=torch.long)

        # Q-Former: query tokens cross-attend to image features, conditioned on instruction
        qf_out = self.base.qformer(
            input_ids      = qformer_input_ids,
            attention_mask = qformer_attention_mask,
            encoder_hidden_states      = image_embeds,
            encoder_attention_mask     = image_attn,
            return_dict    = True,
        )
        # Mean-pool the 32 query token outputs → (B, 768)
        query_features = qf_out.last_hidden_state.mean(dim=1)
        cls_logits     = self.classifier(query_features)

        # ── Full model forward for generation loss ─────────────────────
        gen_loss = None
        if labels is not None:
            gen_out = self.base(
                pixel_values           = pixel_values,
                input_ids              = input_ids,
                attention_mask         = attention_mask,
                qformer_input_ids      = qformer_input_ids,
                qformer_attention_mask = qformer_attention_mask,
                labels                 = labels,
            )
            gen_loss = gen_out.loss

        return cls_logits, gen_loss

    @torch.no_grad()
    def generate_description(self, pixel_values, input_ids, attention_mask,
                              qformer_input_ids, qformer_attention_mask,
                              max_new_tokens: int = MAX_GEN_TOKENS,
                              num_beams: int = 4):
        return self.base.generate(
            pixel_values           = pixel_values,
            input_ids              = input_ids,
            attention_mask         = attention_mask,
            qformer_input_ids      = qformer_input_ids,
            qformer_attention_mask = qformer_attention_mask,
            max_new_tokens         = max_new_tokens,
            num_beams              = num_beams,
            repetition_penalty     = 1.2,
            length_penalty         = 1.0,
            no_repeat_ngram_size   = 3,
        )


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

class GenerationMetrics:
    def __init__(self):
        self._scorer     = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        self._hypotheses: list[str]       = []
        self._references: list[str]       = []
        self._all_refs:   list[list[str]] = []

    def reset(self):
        self._hypotheses.clear()
        self._references.clear()
        self._all_refs.clear()

    def add_batch(self, predictions: list[str], references: list[str],
                  all_references: list[list[str]] | None = None):
        self._hypotheses.extend(predictions)
        self._references.extend(references)
        self._all_refs.extend(all_references if all_references else [[r] for r in references])

    def compute(self) -> dict[str, float]:
        r1, r2, rL, mt = [], [], [], []
        use_multi = len(self._all_refs) == len(self._hypotheses)
        for i, (hyp, ref) in enumerate(zip(self._hypotheses, self._references)):
            sc = self._scorer.score(ref, hyp)
            r1.append(sc["rouge1"].fmeasure)
            r2.append(sc["rouge2"].fmeasure)
            rL.append(sc["rougeL"].fmeasure)
            ref_list = self._all_refs[i] if use_multi else [ref]
            mt.append(meteor_score(
                [nltk.word_tokenize(r.lower()) for r in ref_list],
                nltk.word_tokenize(hyp.lower()),
            ))
        def avg(lst): return round(sum(lst) / len(lst), 4) if lst else 0.0
        return {"rouge1": avg(r1), "rouge2": avg(r2), "rougeL": avg(rL), "meteor": avg(mt)}


# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────

def run_epoch(model, loader, optimizer, cls_criterion, processor, scaler,
              is_train: bool, epoch: int, run_generation: bool = True,
              scheduler=None):
    model.train() if is_train else model.eval()
    total_loss = total_cls = total_gen = 0.0
    all_preds, all_labels, sample_outputs = [], [], []
    gen_metrics = GenerationMetrics()
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        loop = tqdm(loader, desc=f"  Epoch {epoch} [{'Train' if is_train else 'Val  '}]", leave=False)
        if is_train:
            optimizer.zero_grad()

        for step, batch in enumerate(loop):
            (pixel_values, input_ids, attention_mask,
             qformer_input_ids, qformer_attention_mask,
             cls_labels,
             lm_input_ids, lm_attention_mask, labels,
             true_descs, true_all_refs) = batch

            pixel_values    = pixel_values.to(DEVICE)
            input_ids       = input_ids.to(DEVICE)
            attention_mask  = attention_mask.to(DEVICE)
            cls_labels      = cls_labels.to(DEVICE)
            lm_input_ids    = lm_input_ids.to(DEVICE)
            lm_attention_mask = lm_attention_mask.to(DEVICE)
            labels          = labels.to(DEVICE)

            if qformer_input_ids is not None:
                qformer_input_ids      = qformer_input_ids.to(DEVICE)
                qformer_attention_mask = qformer_attention_mask.to(DEVICE)

            with autocast("cuda", enabled=USE_AMP, dtype=torch.bfloat16):
                cls_logits, gen_loss = model(
                    pixel_values           = pixel_values,
                    input_ids              = input_ids,
                    attention_mask         = attention_mask,
                    qformer_input_ids      = qformer_input_ids,
                    qformer_attention_mask = qformer_attention_mask,
                    lm_input_ids           = lm_input_ids,
                    lm_attention_mask      = lm_attention_mask,
                    labels                 = labels,
                )
                cls_loss   = cls_criterion(cls_logits, cls_labels)
                batch_loss = (cls_loss + LAMBDA_GEN * gen_loss) / ACCUMULATION_STEPS

            if is_train:
                scaler.scale(batch_loss).backward()
                if (step + 1) % ACCUMULATION_STEPS == 0 or (step + 1) == len(loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    if scheduler is not None:
                        scheduler.step()
                    if step % 50 == 0:
                        torch.cuda.empty_cache()

            unscaled   = (cls_loss + LAMBDA_GEN * gen_loss).item()
            total_loss += unscaled
            total_cls  += cls_loss.item()
            total_gen  += gen_loss.item()

            preds = cls_logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(cls_labels.cpu().tolist())

            if not is_train and run_generation:
                beams   = 2 if epoch > 0 else 4
                gen_ids = model.generate_description(
                    pixel_values, input_ids, attention_mask,
                    qformer_input_ids, qformer_attention_mask,
                    num_beams=beams,
                )
                gen_texts = [
                    processor.tokenizer.decode(g, skip_special_tokens=True).strip()
                    for g in gen_ids
                ]

                gen_pairs = [(p, r) for p, r in zip(gen_texts, true_descs) if p.strip()]
                if gen_pairs:
                    preds_f, refs_f = zip(*gen_pairs)
                    idxs     = [j for j, p in enumerate(gen_texts) if p.strip()]
                    refs_all = [list(true_all_refs[j]) for j in idxs]
                    gen_metrics.add_batch(list(preds_f), list(refs_f), refs_all)

                if len(sample_outputs) < 6:
                    for pred_txt, true_txt, pc, tc in zip(
                        gen_texts, true_descs, preds, cls_labels.cpu().tolist()
                    ):
                        if len(sample_outputs) < 6:
                            sample_outputs.append({
                                "pred_label": "Fracture" if pc == 1 else "Normal",
                                "true_label": "Fracture" if tc == 1 else "Normal",
                                "pred_desc": pred_txt,
                                "true_desc": true_txt,
                            })

            loop.set_postfix(
                loss=f"{unscaled:.3f}",
                cls=f"{cls_loss.item():.3f}",
                gen=f"{gen_loss.item():.3f}",
            )

    n = len(loader)
    return {
        "loss":           total_loss / n,
        "cls_loss":       total_cls  / n,
        "gen_loss":       total_gen  / n,
        "acc":            accuracy_score(all_labels, all_preds),
        "f1":             f1_score(all_labels, all_preds, average="weighted"),
        "preds":          all_preds,
        "labels":         all_labels,
        "text_scores":    gen_metrics.compute() if (not is_train and run_generation) else {},
        "sample_outputs": sample_outputs,
    }


def _print_gen_scores(scores: dict, label: str = "Val"):
    if not scores:
        return
    print(f"  {label:<4}   ROUGE-1={scores['rouge1']:.4f}  "
          f"ROUGE-2={scores['rouge2']:.4f}  "
          f"ROUGE-L={scores['rougeL']:.4f}  "
          f"METEOR={scores['meteor']:.4f}")


def _print_samples(samples: list[dict], epoch: int):
    print(f"\n  ── Sample Outputs (Epoch {epoch}) ─────────────────────")
    for i, s in enumerate(samples, 1):
        mark = "✓" if s["pred_label"] == s["true_label"] else "✗"
        print(f"  [{i}] {mark}  pred={s['pred_label']:<10} true={s['true_label']}")
        print(f"       pred : {s['pred_desc']}")
        print(f"       true : {s['true_desc']}")
    print()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  InstructBLIP Fracture Classifier + Localizer")
    print(f"  model={MODEL_NAME}")
    print(f"  device={DEVICE}  AMP={USE_AMP}  batch={BATCH_SIZE}  "
          f"accum={ACCUMULATION_STEPS}  effective_batch={BATCH_SIZE*ACCUMULATION_STEPS}")
    print(f"  lr_cls={LR_CLS:.1e}  lr_lora={LR_LORA:.1e}  lambda_gen={LAMBDA_GEN}")
    print("=" * 65 + "\n")

    print("Loading InstructBLIP processor & model …")
    processor = InstructBlipProcessor.from_pretrained(MODEL_NAME)
    model     = InstructBLIPDualHead(MODEL_NAME).to(DEVICE)

    # Pre-tokenize the fixed instruction once — used by collate_fn every batch
    global _INSTR_IDS, _INSTR_LEN
    _instr_enc  = processor.tokenizer([INSTRUCTION], return_tensors="pt", padding=False)
    _INSTR_IDS  = _instr_enc["input_ids"][0]   # (seq_len,)
    _INSTR_LEN  = _INSTR_IDS.shape[0]
    print(f"  Instruction pre-tokenized: {_INSTR_LEN} tokens")
    scaler    = GradScaler("cuda", enabled=USE_SCALER)  # disabled for bfloat16

    cls_trainable   = sum(p.numel() for p in model.classifier.parameters())
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  + classifier head: {cls_trainable:,} params | total trainable: {total_trainable:,}")

    print("\nSplitting data at patient level …\n")
    train_df, val_df, test_df = get_patient_level_splits(TRAIN_CSV, VAL_CSV)

    if DEBUG_MODE:
        train_df = train_df.head(64)
        val_df   = val_df.head(64)
        test_df  = test_df.head(64)

    print("\nComputing sample weights …")
    sample_weights = compute_sample_weights(train_df)
    sampler        = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    _collate       = lambda b: collate_fn(b, processor)

    train_loader = DataLoader(
        FractureDataset(train_df, processor, augment=True),
        batch_size=BATCH_SIZE, sampler=sampler, collate_fn=_collate,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
        prefetch_factor=2,   # pre-load 2 batches ahead per worker
    )
    val_loader = DataLoader(
        FractureDataset(val_df, processor, augment=False),
        batch_size=BATCH_SIZE, shuffle=False, collate_fn=_collate,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
        prefetch_factor=2,
    )
    test_loader = DataLoader(
        FractureDataset(test_df, processor, augment=False),
        batch_size=BATCH_SIZE, shuffle=False, collate_fn=_collate,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
        prefetch_factor=2,
    )

    cls_params  = list(model.classifier.parameters())
    cls_ids     = {id(p) for p in cls_params}
    lora_params = [p for p in model.parameters() if p.requires_grad and id(p) not in cls_ids]

    optimizer = torch.optim.AdamW([
        {"params": cls_params,  "lr": LR_CLS,  "weight_decay": 1e-2},
        {"params": lora_params, "lr": LR_LORA, "weight_decay": 1e-2},
    ])

    steps_per_epoch = len(train_loader) // ACCUMULATION_STEPS
    total_steps     = steps_per_epoch * EPOCHS
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr          = [LR_CLS, LR_LORA],
        total_steps     = total_steps,
        pct_start       = 0.05,
        anneal_strategy = 'cos',
        div_factor      = 10.0,
        final_div_factor = 100.0,
    )

    cls_criterion    = FocalLoss(alpha=0.75, gamma=2.0)
    best_val_f1      = 0.0
    patience_counter = 0

    print(f"\nTraining for up to {EPOCHS} epochs (patience={PATIENCE}) …\n")

    for epoch in range(1, EPOCHS + 1):
        cur_lrs = [pg["lr"] for pg in optimizer.param_groups]
        print(f"Epoch {epoch}/{EPOCHS}  lr_cls={cur_lrs[0]:.2e}  lr_lora={cur_lrs[1]:.2e}")

        do_gen = (epoch % GEN_EVAL_EVERY == 0) or (epoch == EPOCHS)

        tr = run_epoch(model, train_loader, optimizer, cls_criterion, processor, scaler,
                       is_train=True,  epoch=epoch, run_generation=False, scheduler=scheduler)
        vl = run_epoch(model, val_loader,   optimizer, cls_criterion, processor, scaler,
                       is_train=False, epoch=epoch, run_generation=do_gen)

        print(f"  Train  loss={tr['loss']:.4f}  cls={tr['cls_loss']:.4f}  "
              f"gen={tr['gen_loss']:.4f}  acc={tr['acc']*100:.2f}%  f1={tr['f1']:.4f}")
        print(f"  Val    loss={vl['loss']:.4f}  cls={vl['cls_loss']:.4f}  "
              f"gen={vl['gen_loss']:.4f}  acc={vl['acc']*100:.2f}%  f1={vl['f1']:.4f}")

        _print_gen_scores(vl.get("text_scores", {}), "Val")

        if do_gen and vl.get("sample_outputs"):
            _print_samples(vl["sample_outputs"], epoch)

        if vl["f1"] > best_val_f1:
            best_val_f1      = vl["f1"]
            patience_counter = 0
            os.makedirs(SAVE_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pt"))
            print(f"  ✓ Best model saved (val F1={best_val_f1:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{PATIENCE})")
            if patience_counter >= PATIENCE:
                print(f"\n  Early stopping triggered at epoch {epoch}.")
                break
        print()

    print("\n" + "=" * 65)
    print("  FINAL TEST SET EVALUATION")
    print("=" * 65)

    model.load_state_dict(torch.load(
        os.path.join(SAVE_DIR, "best_model.pt"), map_location=DEVICE,
    ))
    test_res = run_epoch(
        model, test_loader, None, cls_criterion, processor, scaler,
        is_train=False, epoch=0, run_generation=True,
    )

    print("\n  Classification Report:")
    print(classification_report(
        test_res["labels"], test_res["preds"],
        target_names=["Normal", "Fracture"],
    ))
    _print_gen_scores(test_res.get("text_scores", {}), "Test")

    if test_res.get("sample_outputs"):
        _print_samples(test_res["sample_outputs"], epoch=0)

    print(f"\nAll artefacts saved → {SAVE_DIR}/")


if __name__ == "__main__":
    main()
