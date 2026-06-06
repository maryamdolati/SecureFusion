

!pip -q install transformers accelerate scikit-learn pandas tqdm

import os, re, gc, json, random, warnings
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, matthews_corrcoef,
    cohen_kappa_score, brier_score_loss, confusion_matrix
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

warnings.filterwarnings("ignore")



CSV_PATH   = "/content/V3_nvd_final_only_V3.csv"
TEXT_MAIN  = "Description"
TARGET_COL = "Exploitability"

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEXT_MODEL_NAME  = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_HIDDEN_SIZE = 384
MAX_LEN          = 384

TEXT_BATCH_SIZE   = 8
TEXT_LR           = 3e-5
TEXT_WEIGHT_DECAY = 0.03
TEXT_DROPOUT      = 0.35

TEACHER_MAX_EPOCHS     = 10
TEACHER_EARLY_PATIENCE = 3

STUDENT_MAX_EPOCHS     = 6
STUDENT_EARLY_PATIENCE = 1

DISTILL_TEMPERATURE = 3.0
ALPHA               = 0.8
MAX_GRAD_NORM       = 1.0

TAB_BATCH = 256
TAB_EARLY_PATIENCE = 8

ROBUST_TEXT_TRAIN = True
TEXT_PERTURB_PROB = 0.15

ROBUST_TAB_TRAIN = True
TAB_PERTURB_PROB = 0.30
TAB_NOISE_STD    = 0.10

ROBUST_META_TRAIN = True

TEST_PERTURB_RATIO = 0.30
TEST_NOISE_RATIO   = 0.30

print("Device:", DEVICE)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(SEED)


def safe_roc_auc(y_true, y_prob):
    try:
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return np.nan

def safe_pr_auc(y_true, y_prob):
    try:
        return float(average_precision_score(y_true, y_prob))
    except Exception:
        return np.nan

def best_threshold(y_true, y_prob):
   
    grid = np.linspace(0.01, 0.99, 99)
    scores = [
        f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        for t in grid
    ]
    idx = int(np.argmax(scores))
    return float(grid[idx]), float(scores[idx])

def compute_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp + 1e-12)
    fpr = fp / (fp + tn + 1e-12)

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": specificity,
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC": safe_roc_auc(y_true, y_prob),
        "PR_AUC": safe_pr_auc(y_true, y_prob),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Brier": brier_score_loss(y_true, y_prob),
        "Kappa": cohen_kappa_score(y_true, y_pred),
        "FPR": fpr,
        "Threshold": threshold,
    }

def print_metrics(name, metrics):
    print(f"\n{name}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")



print("\nLoading CSV:", CSV_PATH)
df = pd.read_csv(CSV_PATH)
df.columns = [re.sub(r"\s+", "_", c).strip() for c in df.columns]

print("Loaded shape:", df.shape)
print("Columns:", df.columns.tolist())

if TARGET_COL not in df.columns:
    raise ValueError(f"{TARGET_COL} not found!")

df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
print("NaNs in Exploitability before cleaning:", df[TARGET_COL].isna().sum())

valid_mask = df[TARGET_COL].isin([0.0, 1.0])
invalid_count = (~valid_mask).sum()
if invalid_count > 0:
    print(f"Removing {invalid_count} rows with non-binary Exploitability values")

df = df[valid_mask].copy()
df = df.dropna(subset=[TARGET_COL]).copy()
df[TARGET_COL] = df[TARGET_COL].astype(int)

print("NaNs in Exploitability after cleaning:", df[TARGET_COL].isna().sum())
print("Class balance:\n", df[TARGET_COL].value_counts(normalize=True).rename("proportion"))


if "PUBLISHEDDATE" not in df.columns:
    for cand in ["Published", "published", "PUBLISHED", "NVD_Published"]:
        if cand in df.columns:
            df["PUBLISHEDDATE"] = df[cand]
            break

if "LASTMODIFIEDDATE" not in df.columns:
    for cand in ["LastModified", "LASTMODIFIED", "Last_Modified", "NVD_LastModified"]:
        if cand in df.columns:
            df["LASTMODIFIEDDATE"] = df[cand]
            break

def parse_cvss_v3(s):
    out = {}
    if not isinstance(s, str) or "CVSS:3" not in s:
        return out
    for p in s.split("/"):
        if ":" in p:
            k, v = p.split(":", 1)
            k, v = k.strip(), v.strip()
            if k in {"AV", "AC", "PR", "UI", "S", "C", "I", "A"}:
                out[f"CVSSv3_{k}"] = v
    return out

def parse_cvss_v2(s):
    out = {}
    if not isinstance(s, str):
        return out
    for p in s.split("/"):
        if ":" in p:
            k, v = p.split(":", 1)
            k, v = k.strip(), v.strip()
            if k in {"AV", "AC", "Au", "C", "I", "A"}:
                out[f"CVSSv2_{k}"] = v
    return out

cvss_v3_cols = []
for cand in ["VectorString", "Vector_v3", "NVD_VectorString", "NVD_Vector_v3"]:
    if cand in df.columns:
        cvss_v3_cols.append(cand)

for vcol in cvss_v3_cols:
    add = pd.DataFrame(list(df[vcol].apply(parse_cvss_v3))).fillna(np.nan)
    for c in add.columns:
        df[c] = add[c]

for vcol in ["Vector_v2_fromNVD", "Vector_v2", "NVD_Vector_v2_fromNVD", "NVD_Vector_v2"]:
    if vcol in df.columns:
        add2 = pd.DataFrame(list(df[vcol].apply(parse_cvss_v2))).fillna(np.nan)
        for c in add2.columns:
            df[c] = add2[c]

for raw in [
    "VectorString", "Vector_v3", "Vector_v2_fromNVD", "Vector_v2",
    "NVD_VectorString", "NVD_Vector_v3", "NVD_Vector_v2_fromNVD", "NVD_Vector_v2"
]:
    if raw in df.columns:
        df.drop(columns=[raw], inplace=True, errors="ignore")

for c in ["PUBLISHEDDATE", "LASTMODIFIEDDATE"]:
    if c in df.columns:
        df[c] = pd.to_datetime(df[c], errors="coerce")

if all(c in df.columns for c in ["PUBLISHEDDATE", "LASTMODIFIEDDATE"]):
    df["age_days"] = (df["LASTMODIFIEDDATE"] - df["PUBLISHEDDATE"]).dt.days
    df["age_days"] = df["age_days"].fillna(df["age_days"].median())

for side in ["PUBLISHEDDATE", "LASTMODIFIEDDATE"]:
    if side in df.columns:
        df[f"{side}_year"]   = df[side].dt.year
        df[f"{side}_month"]  = df[side].dt.month
        df[f"{side}_dow"]    = df[side].dt.dayofweek
        df[f"{side}_dom"]    = df[side].dt.day
        df[f"{side}_isnull"] = df[side].isna().astype(int)

drop_conflict = [
    c for c in df.columns
    if c.lower().startswith("conflict_flag")
    or c.lower() in {"conflict_count", "conflict_any"}
]
df.drop(columns=drop_conflict, inplace=True, errors="ignore")


LEAKAGE_REF_COLS = [
    "HasExploitRef",
    "HasMetasploitRef",
    "HasGithubRef",
    "HasVendorAdvisory",
    "RefTags_all",
]

LEAKY_SCORE_COLS = [
    "ExploitabilityScore",
    "NVD_ExploitabilityScore",
    "Delta_ExploitabilityScore",
    "ImpactScore",
    "NVD_ImpactScore",
    "Delta_ImpactScore",
]

TO_DROP_LEAKAGE = LEAKAGE_REF_COLS + LEAKY_SCORE_COLS
removed_leakage_cols = [c for c in TO_DROP_LEAKAGE if c in df.columns]
df.drop(columns=removed_leakage_cols, inplace=True, errors="ignore")

print("\nRemoved leakage columns:")
print(removed_leakage_cols)

if TEXT_MAIN not in df.columns:
    df[TEXT_MAIN] = ""
df[TEXT_MAIN] = df[TEXT_MAIN].fillna("").astype(str)


df_trainval, df_te = train_test_split(
    df,
    test_size=0.15,
    random_state=SEED,
    stratify=df[TARGET_COL],
)

df_tr, df_va = train_test_split(
    df_trainval,
    test_size=0.2,
    random_state=SEED,
    stratify=df_trainval[TARGET_COL],
)

df_tr = df_tr.reset_index(drop=True)
df_va = df_va.reset_index(drop=True)
df_te = df_te.reset_index(drop=True)

print("\nSplit sizes (train / val / test):", len(df_tr), len(df_va), len(df_te))




NON_FEATURE_COLS = {
    TARGET_COL,
    TEXT_MAIN,
    "ID",
    "PUBLISHEDDATE",
    "LASTMODIFIEDDATE",
}

def pick_cols_all(df_):
    num_cols, cat_cols = [], []
    for c in df_.columns:
        if c in NON_FEATURE_COLS:
            continue
        s = df_[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            continue
        if pd.api.types.is_numeric_dtype(s):
            num_cols.append(c)
        else:
            cat_cols.append(c)
    return num_cols, cat_cols

NUM_COLS, CAT_COLS = pick_cols_all(df_tr)

const_cols = [c for c in NUM_COLS if df_tr[c].nunique(dropna=False) <= 1]
const_cols += [c for c in CAT_COLS if df_tr[c].nunique(dropna=False) <= 1]

NUM_COLS = [c for c in NUM_COLS if c not in const_cols]
CAT_COLS = [c for c in CAT_COLS if c not in const_cols]

print("\nFeature counts after leakage removal:")
print("NUM features:", len(NUM_COLS))
print("CAT features:", len(CAT_COLS))

leakage_check = LEAKAGE_REF_COLS + LEAKY_SCORE_COLS
print("\nLeakage safety check:")
print("Still in dataframe:", [c for c in leakage_check if c in df.columns])
print("Still in NUM_COLS:", [c for c in leakage_check if c in NUM_COLS])
print("Still in CAT_COLS:", [c for c in leakage_check if c in CAT_COLS])

assert len([c for c in leakage_check if c in NUM_COLS]) == 0
assert len([c for c in leakage_check if c in CAT_COLS]) == 0

scaler = StandardScaler()

if NUM_COLS:
    Xn_tr_df = df_tr[NUM_COLS].apply(pd.to_numeric, errors="coerce")
    medians = Xn_tr_df.median(numeric_only=True)
    Xn_tr_df = Xn_tr_df.fillna(medians)
    scaler.fit(Xn_tr_df.values)
else:
    medians = pd.Series(dtype=float)

encoders = {}
for c in CAT_COLS:
    le = LabelEncoder()
    vals = df_tr[c].astype(str).fillna("___NA___").values.tolist()
    vals += ["___UNK___"]
    le.fit(vals)
    encoders[c] = le

def transform_num(df_part):
    if not NUM_COLS:
        return np.zeros((len(df_part), 0), dtype=np.float32)
    X = df_part[NUM_COLS].apply(pd.to_numeric, errors="coerce").fillna(medians)
    return scaler.transform(X.values).astype(np.float32)

def transform_cat(df_part):
    if not CAT_COLS:
        return np.zeros((len(df_part), 0), dtype=np.int64)
    out = np.zeros((len(df_part), len(CAT_COLS)), dtype=np.int64)
    for j, c in enumerate(CAT_COLS):
        le = encoders[c]
        vals = df_part[c].astype(str).fillna("___NA___").values
        known = set(le.classes_)
        mapped = [v if v in known else "___UNK___" for v in vals]
        out[:, j] = le.transform(mapped)
    return out

t_tr = df_tr[TEXT_MAIN].astype(str).values
t_va = df_va[TEXT_MAIN].astype(str).values
t_te = df_te[TEXT_MAIN].astype(str).values

y_tr = df_tr[TARGET_COL].astype(int).values
y_va = df_va[TARGET_COL].astype(int).values
y_te = df_te[TARGET_COL].astype(int).values

Xn_tr, Xn_va, Xn_te = transform_num(df_tr), transform_num(df_va), transform_num(df_te)
Xc_tr, Xc_va, Xc_te = transform_cat(df_tr), transform_cat(df_va), transform_cat(df_te)

print("\nShapes:")
print("Xn_tr:", Xn_tr.shape, "Xc_tr:", Xc_tr.shape)




text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
PAD_ID = text_tokenizer.pad_token_id
UNK_ID = text_tokenizer.unk_token_id if text_tokenizer.unk_token_id is not None else PAD_ID

class TextOnlyDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = list(texts)
        self.labels = list(labels)
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        return self.texts[idx], float(self.labels[idx])

def collate_text_with_labels(batch):
    texts, labels = zip(*batch)
    enc = text_tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=MAX_LEN,
    )
    labels = torch.tensor(labels, dtype=torch.float32)
    return enc["input_ids"], enc["attention_mask"], labels

def make_text_loader(texts, labels, batch_size=TEXT_BATCH_SIZE, shuffle=False):
    ds = TextOnlyDataset(texts, labels)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_text_with_labels,
    )

text_train_loader = make_text_loader(t_tr, y_tr, TEXT_BATCH_SIZE, shuffle=True)
text_val_loader   = make_text_loader(t_va, y_va, TEXT_BATCH_SIZE, shuffle=False)
text_test_loader  = make_text_loader(t_te, y_te, TEXT_BATCH_SIZE, shuffle=False)

def perturb_text_batch(input_ids, attention_mask, perturb_prob=TEXT_PERTURB_PROB):
    if perturb_prob <= 0:
        return input_ids, attention_mask

    input_ids = input_ids.clone()
    attention_mask = attention_mask.clone()

    B = input_ids.size(0)
    device = input_ids.device
    mask = torch.rand(B, device=device) < perturb_prob
    idx = torch.nonzero(mask).squeeze(-1)

    if idx.numel() == 0:
        return input_ids, attention_mask

    idx = idx[torch.randperm(idx.numel())]
    half = idx.numel() // 2

    miss_idx = idx[:half]
    noisy_idx = idx[half:]

    if miss_idx.numel() > 0:
        input_ids[miss_idx] = PAD_ID
        attention_mask[miss_idx] = 0

    if noisy_idx.numel() > 0:
        for i in noisy_idx:
            i = int(i.item())
            valid_pos = (input_ids[i] != PAD_ID).nonzero(as_tuple=False).squeeze(-1)
            if valid_pos.numel() == 0:
                continue
            k = max(1, int(0.30 * valid_pos.numel()))
            chosen = valid_pos[torch.randperm(valid_pos.numel())[:k]]
            input_ids[i, chosen] = UNK_ID

    return input_ids, attention_mask

class TextTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(TEXT_MODEL_NAME)
        self.dropout = nn.Dropout(TEXT_DROPOUT)
        self.classifier = nn.Linear(TEXT_HIDDEN_SIZE, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(cls)).squeeze(-1)
        return logits

class TextStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(TEXT_MODEL_NAME)

        if hasattr(self.bert, "encoder") and hasattr(self.bert.encoder, "layer"):
            layers_to_keep = [0, 2, 3, 5]
            new_layers = nn.ModuleList([
                layer for i, layer in enumerate(self.bert.encoder.layer)
                if i in layers_to_keep
            ])
            self.bert.encoder.layer = new_layers
            self.bert.config.num_hidden_layers = len(layers_to_keep)

        self.dropout = nn.Dropout(TEXT_DROPOUT)
        self.classifier = nn.Linear(TEXT_HIDDEN_SIZE, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(cls)).squeeze(-1)
        return logits

neg_count = max(1, int(np.sum(y_tr == 0)))
pos_count = max(1, int(np.sum(y_tr == 1)))
pos_weight_text = torch.tensor(neg_count / pos_count, dtype=torch.float32, device=DEVICE)

teacher_model = TextTeacher().to(DEVICE)
student_model = TextStudent().to(DEVICE)

criterion_teacher = nn.BCEWithLogitsLoss(pos_weight=pos_weight_text)
criterion_hard    = nn.BCEWithLogitsLoss(pos_weight=pos_weight_text)
criterion_soft    = nn.BCEWithLogitsLoss()

def eval_text_model(model, loader, criterion=None):
    model.eval()
    losses, probs_all, y_all = [], [], []

    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(input_ids, attention_mask)

            if criterion is not None:
                loss = criterion(logits, labels)
                losses.append(loss.item())

            probs = torch.sigmoid(logits)
            probs_all.append(probs.detach().cpu().numpy())
            y_all.append(labels.detach().cpu().numpy())

    probs_all = np.concatenate(probs_all)
    y_all = np.concatenate(y_all)
    preds = (probs_all >= 0.5).astype(int)

    return {
        "loss": float(np.mean(losses)) if losses else np.nan,
        "acc": accuracy_score(y_all, preds),
        "f1": f1_score(y_all, preds, zero_division=0),
        "roc_auc": safe_roc_auc(y_all, probs_all),
        "probs": probs_all,
        "y": y_all,
    }


# ---------------- Train teacher ----------------
optimizer_teacher = torch.optim.AdamW(
    teacher_model.parameters(),
    lr=TEXT_LR,
    weight_decay=TEXT_WEIGHT_DECAY,
)

best_teacher_f1 = -1
teacher_patience = 0
teacher_path = "/content/teacher_tuned_leakfree.pt"

print("\n===== TRAINING TEACHER =====")

for epoch in range(1, TEACHER_MAX_EPOCHS + 1):
    teacher_model.train()
    losses, probs_all, y_all = [], [], []

    for input_ids, attention_mask, labels in tqdm(text_train_loader, desc=f"Teacher epoch {epoch}", leave=False):
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer_teacher.zero_grad()
        logits = teacher_model(input_ids, attention_mask)
        loss = criterion_teacher(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(teacher_model.parameters(), MAX_GRAD_NORM)
        optimizer_teacher.step()

        losses.append(loss.item())
        probs_all.append(torch.sigmoid(logits).detach().cpu().numpy())
        y_all.append(labels.detach().cpu().numpy())

    probs_all = np.concatenate(probs_all)
    y_all = np.concatenate(y_all)
    train_f1 = f1_score(y_all, (probs_all >= 0.5).astype(int), zero_division=0)

    va = eval_text_model(teacher_model, text_val_loader, criterion_teacher)

    print(
        f"[Teacher][Epoch {epoch}] "
        f"train_loss={np.mean(losses):.4f} train_f1={train_f1:.4f} | "
        f"val_f1={va['f1']:.4f} val_roc={va['roc_auc']:.4f}"
    )

    if va["f1"] > best_teacher_f1 + 1e-4:
        best_teacher_f1 = va["f1"]
        teacher_patience = 0
        torch.save(teacher_model.state_dict(), teacher_path)
        print("  saved best teacher")
    else:
        teacher_patience += 1
        if teacher_patience >= TEACHER_EARLY_PATIENCE:
            print("  teacher early stopping")
            break

teacher_model.load_state_dict(torch.load(teacher_path, map_location=DEVICE))
teacher_model.eval()


# ---------------- Train student ----------------
if hasattr(student_model.bert, "encoder") and hasattr(student_model.bert.encoder, "layer"):
    for param in student_model.bert.encoder.layer[:-1].parameters():
        param.requires_grad = False

optimizer_student = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, student_model.parameters()),
    lr=TEXT_LR,
    weight_decay=TEXT_WEIGHT_DECAY,
)

num_steps = STUDENT_MAX_EPOCHS * len(text_train_loader)

scheduler_student = get_linear_schedule_with_warmup(
    optimizer_student,
    num_warmup_steps=int(0.1 * num_steps),
    num_training_steps=num_steps,
)

best_student_f1 = -1
student_patience = 0
student_path = "/content/student_tuned_leakfree.pt"


student_train_accuracies = []
student_val_accuracies = []
student_train_f1s = []
student_val_f1s = []
student_train_losses = []
student_val_losses = []

print("\n===== TRAINING STUDENT =====")

for epoch in range(1, STUDENT_MAX_EPOCHS + 1):
    student_model.train()

    losses = []
    probs_all = []
    y_all = []

    for input_ids, attention_mask, labels in tqdm(
        text_train_loader,
        desc=f"Student epoch {epoch}",
        leave=False
    ):
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        labels = labels.to(DEVICE)

        if ROBUST_TEXT_TRAIN:
            input_ids, attention_mask = perturb_text_batch(
                input_ids,
                attention_mask,
                perturb_prob=TEXT_PERTURB_PROB,
            )

        with torch.no_grad():
            teacher_logits = teacher_model(input_ids, attention_mask)
            teacher_soft = torch.sigmoid(teacher_logits / DISTILL_TEMPERATURE)

        student_logits = student_model(input_ids, attention_mask)

        label_smoothing = 0.10
        smooth_labels = labels * (1 - label_smoothing) + 0.5 * label_smoothing

        hard_loss = criterion_hard(student_logits, smooth_labels)

        soft_loss = criterion_soft(
            student_logits / DISTILL_TEMPERATURE,
            teacher_soft,
        ) * (DISTILL_TEMPERATURE ** 2)

        loss = ALPHA * hard_loss + (1 - ALPHA) * soft_loss

        optimizer_student.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), MAX_GRAD_NORM)
        optimizer_student.step()
        scheduler_student.step()

        losses.append(loss.item())
        probs_all.append(torch.sigmoid(student_logits).detach().cpu().numpy())
        y_all.append(labels.detach().cpu().numpy())

    probs_all = np.concatenate(probs_all)
    y_all = np.concatenate(y_all)

    train_acc = accuracy_score(y_all, (probs_all >= 0.5).astype(int))
    train_f1 = f1_score(y_all, (probs_all >= 0.5).astype(int), zero_division=0)
    train_loss = float(np.mean(losses))

    va = eval_text_model(student_model, text_val_loader, criterion_hard)

    student_train_accuracies.append(train_acc)
    student_val_accuracies.append(va["acc"])
    student_train_f1s.append(train_f1)
    student_val_f1s.append(va["f1"])
    student_train_losses.append(train_loss)
    student_val_losses.append(va["loss"])

    print(
        f"[Student][Epoch {epoch}] "
        f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_f1={train_f1:.4f} | "
        f"val_loss={va['loss']:.4f} val_acc={va['acc']:.4f} val_f1={va['f1']:.4f} val_roc={va['roc_auc']:.4f}"
    )

    if va["f1"] > best_student_f1 + 1e-4:
        best_student_f1 = va["f1"]
        student_patience = 0
        torch.save(student_model.state_dict(), student_path)
        print("  saved best student")
    else:
        student_patience += 1

        if student_patience >= STUDENT_EARLY_PATIENCE:
            print("  student early stopping")
            break

student_model.load_state_dict(torch.load(student_path, map_location=DEVICE))
student_model.eval()

def predict_text_probs(model, texts, labels, batch_size=TEXT_BATCH_SIZE):
    loader = make_text_loader(texts, labels, batch_size=batch_size, shuffle=False)
    out = eval_text_model(model, loader, criterion_hard)
    return out["probs"]

print("\nGenerating text probabilities...")
probs_text_va = predict_text_probs(student_model, t_va, y_va)
probs_text_te = predict_text_probs(student_model, t_te, y_te)

thr_text, _ = best_threshold(y_va, probs_text_va)
text_clean_metrics = compute_metrics(y_te, probs_text_te, thr_text)
print_metrics("TextGuard-Lite clean test", text_clean_metrics)



class TabOnlyDS(Dataset):
    def __init__(self, xnum, xcat, y):
        self.xnum = xnum.astype(np.float32)
        self.xcat = xcat.astype(np.int64)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "num": torch.tensor(self.xnum[idx], dtype=torch.float32),
            "cat": torch.tensor(self.xcat[idx], dtype=torch.long),
            "y": torch.tensor(self.y[idx], dtype=torch.float32),
        }

tab_tr_loader = DataLoader(TabOnlyDS(Xn_tr, Xc_tr, y_tr), batch_size=TAB_BATCH, shuffle=True)
tab_va_loader = DataLoader(TabOnlyDS(Xn_va, Xc_va, y_va), batch_size=TAB_BATCH, shuffle=False)
tab_te_loader = DataLoader(TabOnlyDS(Xn_te, Xc_te, y_te), batch_size=TAB_BATCH, shuffle=False)

cat_cards = [len(encoders[c].classes_) for c in CAT_COLS]

class TabTransformer(nn.Module):
    def __init__(
        self,
        num_numerical,
        cat_cardinalities,
        d_model=128,
        n_heads=4,
        n_layers=2,
        dropout=0.2,
    ):
        super().__init__()

        self.num_proj = nn.Linear(num_numerical, d_model) if num_numerical > 0 else None
        self.embs = nn.ModuleList([nn.Embedding(card, d_model) for card in cat_cardinalities])
        self.cls = nn.Parameter(torch.randn(1, 1, d_model))

        n_tokens = 1 + len(self.embs) + (1 if num_numerical > 0 else 0)
        self.col_emb = nn.Embedding(max(n_tokens, 1), d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, xnum, xcat):
        if xnum is not None and xnum.numel() > 0:
            B = xnum.size(0)
        else:
            B = xcat.size(0)

        tokens = [self.cls.expand(B, -1, -1)]

        if self.embs and xcat is not None and xcat.shape[1] > 0:
            for j, emb in enumerate(self.embs):
                tokens.append(emb(xcat[:, j]).unsqueeze(1))

        if self.num_proj is not None and xnum is not None and xnum.shape[1] > 0:
            tokens.append(self.num_proj(xnum).unsqueeze(1))

        X = torch.cat(tokens, dim=1)
        idx = torch.arange(X.size(1), device=X.device).unsqueeze(0)
        X = X + self.col_emb(idx)

        X = self.encoder(X)
        X = self.norm(X)

        return X[:, 0, :]

class TabOnlyModel(nn.Module):
    def __init__(self, n_num, cat_cards, d_model=128, n_heads=4, n_layers=2, dropout=0.2, head_dropout=0.3):
        super().__init__()
        self.backbone = TabTransformer(
            n_num,
            cat_cards,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )

        self.head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(128, 1),
        )

    def forward(self, xnum, xcat):
        h = self.backbone(xnum, xcat)
        return self.head(h).squeeze(-1)


def train_one_tab_config(cfg):
    seed_everything(SEED)

    model = TabOnlyModel(
        Xn_tr.shape[1],
        cat_cards,
        d_model=cfg["d_model"],
        n_heads=cfg["heads"],
        n_layers=cfg["layers"],
        dropout=cfg["dropout"],
        head_dropout=cfg["head_dropout"],
    ).to(DEVICE)

    neg_tab = max(1, int(np.sum(y_tr == 0)))
    pos_tab = max(1, int(np.sum(y_tr == 1)))

    pos_weight_tab = torch.tensor(
        neg_tab / pos_tab,
        dtype=torch.float32,
        device=DEVICE
    )

    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tab)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"]
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_f1": [],
        "val_f1": [],
        "val_roc_auc": [],
    }

    def run_epoch(loader, train=True):
        if train:
            model.train()
        else:
            model.eval()

        losses = []
        logits_all = []
        y_all = []

        for b in loader:
            xn = b["num"].to(DEVICE)
            xc = b["cat"].to(DEVICE)
            y = b["y"].to(DEVICE)

            if train and ROBUST_TAB_TRAIN:
                xn = xn.clone()
                xc = xc.clone()

                B = xn.size(0)
                mask = torch.rand(B, device=xn.device) < TAB_PERTURB_PROB
                idx = torch.nonzero(mask).squeeze(-1)

                if idx.numel() > 0:
                    idx = idx[torch.randperm(idx.numel())]
                    half = idx.numel() // 2

                    miss_idx = idx[:half]
                    noisy_idx = idx[half:]

                    if miss_idx.numel() > 0:
                        if xn.shape[1] > 0:
                            xn[miss_idx] = 0.0

                        if xc.shape[1] > 0:
                            xc[miss_idx] = 0

                    if noisy_idx.numel() > 0 and xn.shape[1] > 0:
                        xn[noisy_idx] = (
                            xn[noisy_idx]
                            + torch.randn_like(xn[noisy_idx]) * TAB_NOISE_STD
                        )

            with torch.set_grad_enabled(train):
                logits = model(xn, xc)
                loss = crit(logits, y)

                if train:
                    optim.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        MAX_GRAD_NORM
                    )
                    optim.step()

            losses.append(loss.item())
            logits_all.append(logits.detach().cpu())
            y_all.append(y.detach().cpu())

        logits_all = torch.cat(logits_all)
        y_all = torch.cat(y_all).numpy()

        probs = torch.sigmoid(logits_all).numpy()
        preds = (probs >= 0.5).astype(int)

        return {
            "loss": float(np.mean(losses)),
            "acc": accuracy_score(y_all, preds),
            "f1": f1_score(y_all, preds, zero_division=0),
            "roc_auc": safe_roc_auc(y_all, probs),
            "probs": probs,
            "y": y_all,
        }

    best_val_f1 = -1.0
    best_path = f"/content/tab_best_{cfg['name']}.pt"
    patience = 0

    print(f"\n===== TRAINING TAB CONFIG: {cfg['name']} =====")
    print(cfg)

    for ep in range(1, cfg["epochs"] + 1):
        tr = run_epoch(tab_tr_loader, train=True)
        va = run_epoch(tab_va_loader, train=False)


        history["train_loss"].append(tr["loss"])
        history["val_loss"].append(va["loss"])
        history["train_acc"].append(tr["acc"])
        history["val_acc"].append(va["acc"])
        history["train_f1"].append(tr["f1"])
        history["val_f1"].append(va["f1"])
        history["val_roc_auc"].append(va["roc_auc"])

        print(
            f"[{cfg['name']}][Ep {ep}] "
            f"train_loss={tr['loss']:.4f} train_acc={tr['acc']:.4f} train_f1={tr['f1']:.4f} | "
            f"val_loss={va['loss']:.4f} val_acc={va['acc']:.4f} val_f1={va['f1']:.4f} val_roc={va['roc_auc']:.4f}"
        )

        if va["f1"] > best_val_f1 + 1e-4:
            best_val_f1 = va["f1"]
            patience = 0
            torch.save(model.state_dict(), best_path)
            print("  saved best tab")
        else:
            patience += 1

            if patience >= TAB_EARLY_PATIENCE:
                print("  tab early stopping")
                break

    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    model.eval()

    va = run_epoch(tab_va_loader, train=False)
    te = run_epoch(tab_te_loader, train=False)

    thr_tab, val_f1_thr = best_threshold(y_va, va["probs"])
    test_metrics = compute_metrics(y_te, te["probs"], thr_tab)

    best_epoch_by_val_f1 = int(np.argmax(history["val_f1"]) + 1)
    best_epoch_by_val_acc = int(np.argmax(history["val_acc"]) + 1)

    return {
        "cfg": cfg,
        "model_path": best_path,
        "val_probs": va["probs"],
        "test_probs": te["probs"],
        "thr": thr_tab,
        "val_f1_thr": val_f1_thr,
        "test_metrics": test_metrics,
        "history": history,
        "best_epoch_by_val_f1": best_epoch_by_val_f1,
        "best_epoch_by_val_acc": best_epoch_by_val_acc,
    }


TAB_CONFIGS = [
    {
        "name": "base_d128_l2_do02_hd03_lr1e3",
        "d_model": 128,
        "heads": 4,
        "layers": 2,
        "dropout": 0.20,
        "head_dropout": 0.30,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "epochs": 40,
    },
    {
        "name": "lowdrop_d128_l2_do01_hd02_lr1e3",
        "d_model": 128,
        "heads": 4,
        "layers": 2,
        "dropout": 0.10,
        "head_dropout": 0.20,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "epochs": 45,
    },
    {
        "name": "lowdrop_d128_l3_do01_hd02_lr5e4",
        "d_model": 128,
        "heads": 4,
        "layers": 3,
        "dropout": 0.10,
        "head_dropout": 0.20,
        "lr": 5e-4,
        "weight_decay": 1e-4,
        "epochs": 50,
    },
    {
        "name": "wide_d192_l2_do01_hd02_lr5e4",
        "d_model": 192,
        "heads": 4,
        "layers": 2,
        "dropout": 0.10,
        "head_dropout": 0.20,
        "lr": 5e-4,
        "weight_decay": 1e-4,
        "epochs": 50,
    },
    {
        "name": "wide_d192_l3_do015_hd02_lr5e4",
        "d_model": 192,
        "heads": 4,
        "layers": 3,
        "dropout": 0.15,
        "head_dropout": 0.20,
        "lr": 5e-4,
        "weight_decay": 1e-4,
        "epochs": 50,
    },
    {
        "name": "d256_l2_h8_do015_hd02_lr5e4",
        "d_model": 256,
        "heads": 8,
        "layers": 2,
        "dropout": 0.15,
        "head_dropout": 0.20,
        "lr": 5e-4,
        "weight_decay": 1e-4,
        "epochs": 50,
    },
]


tab_trials = []

for cfg in TAB_CONFIGS:
    try:
        trial = train_one_tab_config(cfg)
        tab_trials.append(trial)

        print_metrics(
            f"TAB TEST RESULT: {cfg['name']}",
            trial["test_metrics"]
        )

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        print("FAILED CONFIG:", cfg["name"])
        print("ERROR:", e)


if len(tab_trials) == 0:
    raise RuntimeError("No TabShield-Net configuration was trained successfully.")


trial_rows = []

for trl in tab_trials:
    m = trl["test_metrics"]

    row = {
        "config": trl["cfg"]["name"],
        "best_epoch_by_val_f1": trl["best_epoch_by_val_f1"],
        "best_epoch_by_val_acc": trl["best_epoch_by_val_acc"],
        "val_f1_thresholded": trl["val_f1_thr"],
        "threshold": trl["thr"],
        "test_F1": m["F1"],
        "test_ROC_AUC": m["ROC_AUC"],
        "test_PR_AUC": m["PR_AUC"],
        "test_MCC": m["MCC"],
        "test_Brier": m["Brier"],
        "test_FPR": m["FPR"],
    }

    trial_rows.append(row)


tab_tuning_df = pd.DataFrame(trial_rows).sort_values(
    ["val_f1_thresholded", "test_F1"],
    ascending=False
).reset_index(drop=True)

print("\n================ TAB TUNING TRIALS ================")
display(tab_tuning_df)

tab_tuning_df.to_csv(
    "/content/tab_tuning_trials_leakage_free.csv",
    index=False
)




best_config_name = tab_tuning_df.iloc[0]["config"]

best_tab_trial = [
    t for t in tab_trials
    if t["cfg"]["name"] == best_config_name
][0]

probs_tab_va = best_tab_trial["val_probs"]
probs_tab_te = best_tab_trial["test_probs"]
thr_tab = best_tab_trial["thr"]
tab_clean_metrics = best_tab_trial["test_metrics"]




tab_train_accuracies = best_tab_trial["history"]["train_acc"]
tab_val_accuracies = best_tab_trial["history"]["val_acc"]

tab_train_f1s = best_tab_trial["history"]["train_f1"]
tab_val_f1s = best_tab_trial["history"]["val_f1"]

tab_train_losses = best_tab_trial["history"]["train_loss"]
tab_val_losses = best_tab_trial["history"]["val_loss"]

best_ep_tab_acc = int(np.argmax(tab_val_accuracies) + 1)
best_ep_tab_f1 = int(np.argmax(tab_val_f1s) + 1)


print("\nBEST TAB CONFIG:")
print(best_tab_trial["cfg"])

print("\nBEST TAB EPOCHS:")
print("Best epoch by Val Accuracy:", best_ep_tab_acc)
print("Best epoch by Val F1:", best_ep_tab_f1)

print_metrics(
    "Best TabShield-Net clean test",
    tab_clean_metrics
)


def make_meta_train(X, y, robust=True):
    if not robust:
        return X, y

    X_full = X.copy()
    y_full = y.copy()

    X_miss_text = X.copy()
    X_miss_text[:, 0] = 0.5

    X_miss_tab = X.copy()
    X_miss_tab[:, 1] = 0.5

    X_aug = np.concatenate([X_full, X_miss_text, X_miss_tab], axis=0)
    y_aug = np.concatenate([y_full, y_full, y_full], axis=0)

    return X_aug, y_aug

fusion_trials = []

X_meta_va_clean = np.vstack([probs_text_va, probs_tab_va]).T
X_meta_te_clean = np.vstack([probs_text_te, probs_tab_te]).T

for robust_meta in [True, False]:
    for C in [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]:
        for class_weight in [None, "balanced"]:

            X_train_meta, y_train_meta = make_meta_train(
                X_meta_va_clean,
                y_va,
                robust=robust_meta,
            )

            meta = LogisticRegression(
                max_iter=2000,
                random_state=SEED,
                C=C,
                class_weight=class_weight,
                solver="lbfgs",
            )

            meta.fit(X_train_meta, y_train_meta)

            p_va = meta.predict_proba(X_meta_va_clean)[:, 1]
            thr_fusion, val_f1 = best_threshold(y_va, p_va)

            p_te = meta.predict_proba(X_meta_te_clean)[:, 1]
            test_metrics = compute_metrics(y_te, p_te, thr_fusion)

            fusion_trials.append({
                "robust_meta": robust_meta,
                "C": C,
                "class_weight": str(class_weight),
                "threshold": thr_fusion,
                "val_f1": val_f1,
                "test_F1": test_metrics["F1"],
                "test_ROC_AUC": test_metrics["ROC_AUC"],
                "test_PR_AUC": test_metrics["PR_AUC"],
                "test_MCC": test_metrics["MCC"],
                "test_Brier": test_metrics["Brier"],
                "test_FPR": test_metrics["FPR"],
                "meta_model": meta,
                "test_probs": p_te,
                "test_metrics": test_metrics,
            })

fusion_tuning_df = pd.DataFrame([
    {k: v for k, v in r.items() if k not in ["meta_model", "test_probs", "test_metrics"]}
    for r in fusion_trials
]).sort_values(["val_f1", "test_F1"], ascending=False)

print("\n================ FUSION TUNING TRIALS ================")
display(fusion_tuning_df.head(20))

fusion_tuning_df.to_csv("/content/fusion_tuning_trials_leakage_free.csv", index=False)

best_fusion_row = fusion_tuning_df.iloc[0]
best_fusion = None
for r in fusion_trials:
    if (
        r["robust_meta"] == bool(best_fusion_row["robust_meta"])
        and float(r["C"]) == float(best_fusion_row["C"])
        and str(r["class_weight"]) == str(best_fusion_row["class_weight"])
        and abs(float(r["threshold"]) - float(best_fusion_row["threshold"])) < 1e-12
    ):
        best_fusion = r
        break

decision_fusion = best_fusion["meta_model"]
thr_fusion = best_fusion["threshold"]
probs_fusion_te = best_fusion["test_probs"]
fusion_clean_metrics = best_fusion["test_metrics"]

print("\nBEST FUSION CONFIG:")
print({
    "robust_meta": best_fusion["robust_meta"],
    "C": best_fusion["C"],
    "class_weight": best_fusion["class_weight"],
    "threshold": thr_fusion,
})
print_metrics("SecureFusion-X tuned clean test", fusion_clean_metrics)




def corrupt_texts_missing(texts, ratio=0.30, seed=42):
    rng = np.random.default_rng(seed)
    texts_corrupt = np.array(texts, dtype=object).copy()
    n = len(texts_corrupt)
    k = int(round(ratio * n))
    if k <= 0:
        return texts_corrupt
    idx = rng.choice(n, size=k, replace=False)
    texts_corrupt[idx] = ""
    return texts_corrupt

def corrupt_texts_noisy(texts, ratio=0.30, token_noise_ratio=0.30, seed=42):
    rng = np.random.default_rng(seed)
    texts_corrupt = np.array(texts, dtype=object).copy()
    n = len(texts_corrupt)
    k_docs = int(round(ratio * n))
    if k_docs <= 0:
        return texts_corrupt

    idx = rng.choice(n, size=k_docs, replace=False)

    for i in idx:
        tokens = str(texts_corrupt[i]).split()
        if len(tokens) == 0:
            continue
        k_tokens = max(1, int(round(token_noise_ratio * len(tokens))))
        chosen = rng.choice(len(tokens), size=k_tokens, replace=False)
        for j in chosen:
            tokens[j] = "[UNK]"
        texts_corrupt[i] = " ".join(tokens)

    return texts_corrupt

t_te_missing = corrupt_texts_missing(
    t_te,
    ratio=TEST_PERTURB_RATIO,
    seed=SEED,
)

t_te_noisy = corrupt_texts_noisy(
    t_te,
    ratio=TEST_PERTURB_RATIO,
    token_noise_ratio=TEST_NOISE_RATIO,
    seed=SEED,
)

probs_text_missing = predict_text_probs(student_model, t_te_missing, y_te)
probs_text_noisy   = predict_text_probs(student_model, t_te_noisy, y_te)

probs_tab_missing = probs_tab_te.copy()
probs_tab_noisy   = probs_tab_te.copy()

X_meta_missing = np.vstack([probs_text_missing, probs_tab_missing]).T
X_meta_noisy   = np.vstack([probs_text_noisy, probs_tab_noisy]).T

probs_fusion_missing = decision_fusion.predict_proba(X_meta_missing)[:, 1]
probs_fusion_noisy   = decision_fusion.predict_proba(X_meta_noisy)[:, 1]

text_missing_metrics = compute_metrics(y_te, probs_text_missing, thr_text)
text_noisy_metrics   = compute_metrics(y_te, probs_text_noisy, thr_text)

tab_missing_metrics = compute_metrics(y_te, probs_tab_missing, thr_tab)
tab_noisy_metrics   = compute_metrics(y_te, probs_tab_noisy, thr_tab)

fusion_missing_metrics = compute_metrics(y_te, probs_fusion_missing, thr_fusion)
fusion_noisy_metrics   = compute_metrics(y_te, probs_fusion_noisy, thr_fusion)

print_metrics("TextGuard-Lite 30% missing text tuned leakage-free", text_missing_metrics)
print_metrics("TextGuard-Lite 30% noisy text tuned leakage-free", text_noisy_metrics)
print_metrics("SecureFusion-X 30% missing text tuned leakage-free", fusion_missing_metrics)
print_metrics("SecureFusion-X 30% noisy text tuned leakage-free", fusion_noisy_metrics)



rows = []

def add_row(model, setting, metrics):
    row = {
        "Model": model,
        "Setting": setting,
    }
    row.update(metrics)
    rows.append(row)

add_row("TextGuard-Lite", "Clean", text_clean_metrics)
add_row("TextGuard-Lite", "30% Missing Text", text_missing_metrics)
add_row("TextGuard-Lite", "30% Noisy Text", text_noisy_metrics)

add_row("TabShield-Net", "Clean", tab_clean_metrics)
add_row("TabShield-Net", "30% Missing Text", tab_missing_metrics)
add_row("TabShield-Net", "30% Noisy Text", tab_noisy_metrics)

add_row("SecureFusion-X Decision-level", "Clean", fusion_clean_metrics)
add_row("SecureFusion-X Decision-level", "30% Missing Text", fusion_missing_metrics)
add_row("SecureFusion-X Decision-level", "30% Noisy Text", fusion_noisy_metrics)

results_df = pd.DataFrame(rows)

display_cols = [
    "Model", "Setting",
    "Accuracy", "Balanced_Accuracy", "Precision", "Recall", "Specificity",
    "F1", "ROC_AUC", "PR_AUC", "MCC", "Brier", "Kappa", "FPR", "Threshold"
]

results_df = results_df[display_cols]

for c in results_df.columns:
    if c not in ["Model", "Setting"]:
        results_df[c] = results_df[c].astype(float).round(4)

print("\n================ TUNED LEAKAGE-FREE FINAL RESULTS ================")
display(results_df)

results_path = "/content/tuned_leakage_free_securefusion_results.csv"
results_df.to_csv(results_path, index=False)
print("Saved:", results_path)

feature_report = {
    "removed_leakage_columns": removed_leakage_cols,
    "num_features": NUM_COLS,
    "cat_features": CAT_COLS,
    "best_tab_config": best_tab_trial["cfg"],
    "best_fusion_config": {
        "robust_meta": best_fusion["robust_meta"],
        "C": best_fusion["C"],
        "class_weight": best_fusion["class_weight"],
        "threshold": thr_fusion,
    },
}

feature_report_path = "/content/tuned_leakage_free_feature_report.json"
with open(feature_report_path, "w", encoding="utf-8") as f:
    json.dump(feature_report, f, indent=2, ensure_ascii=False)

print("Saved:", feature_report_path)

print("\nFinal leakage check:")
print("Still in dataframe:", [c for c in leakage_check if c in df.columns])
print("Still in NUM_COLS:", [c for c in leakage_check if c in NUM_COLS])
print("Still in CAT_COLS:", [c for c in leakage_check if c in CAT_COLS])

print("\nUse these files:")
print("- /content/tuned_leakage_free_securefusion_results.csv")
print("- /content/tab_tuning_trials_leakage_free.csv")
print("- /content/fusion_tuning_trials_leakage_free.csv")
print("- /content/tuned_leakage_free_feature_report.json")
