# =============================================================================
# module3_detector.py — Deep Learning Logo Detection with ResNet + OpenCV
# =============================================================================
# This module has TWO detection methods:
#
#   Method A — PyTorch ResNet-50 (Deep Learning)
#     • Loads a pretrained ResNet-50 and fine-tunes the final layer
#       to distinguish "has IITM logo" vs "no IITM logo"
#     • Training uses images from logo_samples/ (your own collected samples)
#     • At inference, each screenshot and downloaded image is scored
#
#   Method B — OpenCV Template Matching (Classical CV)
#     • Uses the official IITM logo as a template
#     • Slides it across each image looking for a matching region
#     • Fast and interpretable — good backup / cross-check
#
# The final confidence score is the MAX of both methods.
# =============================================================================

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
from PIL import Image
import pandas as pd
from pathlib import Path
import json

import config   # Central config


# ─────────────────────────────────────────────────────────────────────────────
# Section A — PyTorch Deep Learning Model
# ─────────────────────────────────────────────────────────────────────────────

class LogoDataset(Dataset):
    """
    Custom dataset for logo classification.

    Expected folder structure under logo_samples/:
        logo_samples/
            positive/    ← images that CONTAIN the IITM logo
            negative/    ← images that do NOT contain the IITM logo
    """

    def __init__(self, root_dir: str, transform=None):
        self.samples: list[tuple[str, int]] = []
        self.transform = transform

        positive_dir = os.path.join(root_dir, "positive")
        negative_dir = os.path.join(root_dir, "negative")

        # Label 1 = has IITM logo, Label 0 = does not have logo
        for label, folder in [(1, positive_dir), (0, negative_dir)]:
            if not os.path.isdir(folder):
                print(f"  WARNING: Folder not found — {folder}")
                continue
            for fname in os.listdir(folder):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    self.samples.append(
                        (os.path.join(folder, fname), label)
                    )

        print(f"  Dataset loaded: {len(self.samples)} images "
              f"({sum(1 for _,l in self.samples if l==1)} positive, "
              f"{sum(1 for _,l in self.samples if l==0)} negative)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            # If the image is corrupt, return a blank image
            img = Image.new("RGB", (config.IMG_SIZE, config.IMG_SIZE))

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)


def build_model() -> nn.Module:
    """
    Build a ResNet-50 model for binary classification (logo / no logo).
    We freeze all layers except the final fully-connected head.
    This is called 'transfer learning' — we reuse ImageNet weights.
    """
    # Load pretrained ResNet-50
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Freeze all pretrained layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final FC layer with our binary classifier
    num_features = model.fc.in_features   # 2048 for ResNet-50
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2),               # 2 classes: logo / no-logo
    )

    return model


def get_transforms(augment: bool = False):
    """
    Return image transforms.
    augment=True → for training (adds random flips/crops for variety)
    augment=False → for inference (just resize and normalise)
    """
    # ImageNet normalisation values (standard for pretrained models)
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if augment:
        return transforms.Compose([
            transforms.Resize((config.IMG_SIZE + 32, config.IMG_SIZE + 32)),
            transforms.RandomCrop(config.IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


def train_model(model_save_path: str = None) -> nn.Module:
    """
    Train the ResNet model on the logo_samples/ dataset.
    Saves the trained weights to disk.

    Returns the trained model.
    """
    print("\n" + "─" * 55)
    print("Training Deep Learning Model (ResNet-50) ...")
    print("─" * 55)

    if model_save_path is None:
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        model_save_path = os.path.join(config.MODEL_DIR, "logo_detector.pth")

    # ── Dataset ────────────────────────────────────────────────────────────
    dataset = LogoDataset(
        root_dir=config.LOGO_SAMPLES_DIR,
        transform=get_transforms(augment=True),
    )

    if len(dataset) == 0:
        print("ERROR: No training images found. Please add images to:")
        print(f"  {config.LOGO_SAMPLES_DIR}/positive/  ← images with IITM logo")
        print(f"  {config.LOGO_SAMPLES_DIR}/negative/  ← images without logo")
        raise RuntimeError("Empty dataset")

    # Train/validation split
    n_train = int(len(dataset) * config.TRAIN_SPLIT)
    n_val   = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    # Override validation transform (no augmentation for evaluation)
    val_ds.dataset.transform = get_transforms(augment=False)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE, shuffle=False)

    # ── Model, loss, optimiser ─────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")

    model = build_model().to(device)

    criterion = nn.CrossEntropyLoss()
    # Only optimise the unfrozen FC head
    optimizer = optim.Adam(model.fc.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_val_acc = 0.0

    # ── Training loop ──────────────────────────────────────────────────────
    for epoch in range(1, config.EPOCHS + 1):
        # --- Training phase ---
        model.train()
        train_loss, train_correct = 0.0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item() * images.size(0)
            preds          = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()

        # --- Validation phase ---
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds   = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()

        train_acc = train_correct / n_train
        val_acc   = val_correct   / n_val
        scheduler.step()

        print(f"  Epoch {epoch:02d}/{config.EPOCHS}  "
              f"Loss: {train_loss/n_train:.4f}  "
              f"Train Acc: {train_acc:.3f}  Val Acc: {val_acc:.3f}")

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"    → New best model saved (val acc: {val_acc:.3f})")

    print(f"\n✓ Training complete. Best val accuracy: {best_val_acc:.3f}")
    print(f"  Model saved to: {model_save_path}")

    # Reload best weights before returning
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    return model


def load_trained_model(model_path: str = None) -> tuple[nn.Module, str]:
    """
    Load a previously trained model from disk.
    Returns (model, device_string).
    """
    if model_path is None:
        model_path = os.path.join(config.MODEL_DIR, "logo_detector.pth")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = build_model()

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"  ✓ Loaded model weights from: {model_path}")
    else:
        print(f"  WARNING: No trained model found at {model_path}")
        print("  Run train_model() first, or the DL detector will return 0.0")

    model.to(device)
    model.eval()
    return model, device


def predict_image_dl(model: nn.Module, device: str, image_path: str) -> float:
    """
    Run the trained ResNet model on a single image.

    Returns
    -------
    float — confidence that the image contains the IITM logo (0.0–1.0)
    """
    transform = get_transforms(augment=False)
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        return 0.0

    img_tensor = transform(img).unsqueeze(0).to(device)   # add batch dim

    with torch.no_grad():
        logits = model(img_tensor)                         # shape (1, 2)
        probs  = torch.softmax(logits, dim=1)              # convert to probs
        confidence = probs[0, 1].item()                    # class 1 = logo

    return confidence


# ─────────────────────────────────────────────────────────────────────────────
# Section B — OpenCV Template Matching
# ─────────────────────────────────────────────────────────────────────────────

def load_templates(template_dir: str = "logo_samples/positive") -> list[np.ndarray]:
    """
    Load all images from the positive/ folder as OpenCV templates.
    We use multiple templates at different scales for robustness.
    """
    templates: list[np.ndarray] = []

    if not os.path.isdir(template_dir):
        print(f"  WARNING: Template directory not found: {template_dir}")
        return templates

    for fname in os.listdir(template_dir):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(template_dir, fname)
            tpl  = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if tpl is not None:
                templates.append(tpl)

    print(f"  Loaded {len(templates)} OpenCV templates.")
    return templates


def template_match_score(image_path: str, templates: list[np.ndarray]) -> float:
    """
    Run OpenCV template matching on an image against all templates.

    Uses TM_CCOEFF_NORMED which returns values in [-1, 1].
    We try the template at multiple scales to handle resized logos.

    Returns
    -------
    float — best normalised match score found (0.0–1.0)
    """
    if not templates:
        return 0.0

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0

    best_score = 0.0
    scales = [0.5, 0.75, 1.0, 1.25, 1.5]   # check these resize scales

    for template in templates:
        th, tw = template.shape[:2]

        for scale in scales:
            # Resize template to simulate logo at different sizes on the page
            new_w = int(tw * scale)
            new_h = int(th * scale)
            if new_w < 10 or new_h < 10:   # too small to match
                continue

            # Only resize if image is bigger than (scaled) template
            if img.shape[0] < new_h or img.shape[1] < new_w:
                continue

            tpl_resized = cv2.resize(template, (new_w, new_h))

            # Run template matching
            result = cv2.matchTemplate(img, tpl_resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)

            if max_val > best_score:
                best_score = max_val

    # Clip to [0, 1] range (TM_CCOEFF_NORMED can sometimes exceed 1 slightly)
    return float(np.clip(best_score, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Section C — Combined Detection Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def detect_logo_in_image(
    image_path: str,
    dl_model: nn.Module,
    device: str,
    templates: list[np.ndarray],
) -> dict:
    """
    Run both detection methods on one image.

    Returns
    -------
    dict with keys:
        dl_score       — deep learning confidence (0–1)
        tm_score       — template matching score (0–1)
        final_score    — max of both (our best estimate)
        logo_detected  — True/False based on CONFIDENCE_THRESH
    """
    dl_score = predict_image_dl(dl_model, device, image_path)
    tm_score = template_match_score(image_path, templates)

    # Take the maximum — if either method says "logo found", we flag it
    final_score = max(dl_score, tm_score)

    return {
        "dl_score":      round(dl_score,    4),
        "tm_score":      round(tm_score,    4),
        "final_score":   round(final_score, 4),
        "logo_detected": final_score >= config.CONFIDENCE_THRESH,
    }


def classify_risk(score: float) -> str:
    """Map a confidence score to a risk level string."""
    if score >= config.RISK_HIGH:
        return "HIGH"
    elif score >= config.RISK_MEDIUM:
        return "MEDIUM"
    else:
        return "LOW"


def run_detection(scraped_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run logo detection on all scraped results.

    Parameters
    ----------
    scraped_df : DataFrame output from Module 2 (must have 'screenshot', 'images' cols)

    Returns
    -------
    DataFrame with detection scores and risk levels added
    """
    print("\n" + "=" * 65)
    print("MODULE 3 — DETECTION: Running Logo Detection")
    print("=" * 65)

    # ── Load models ────────────────────────────────────────────────────────
    print("\n[1/2] Loading models ...")
    dl_model, device = load_trained_model()
    templates        = load_templates(os.path.join(config.LOGO_SAMPLES_DIR, "positive"))

    if not templates:
        print("  NOTE: No templates found. OpenCV method will score 0.0 for all.")

    # ── Run detection ──────────────────────────────────────────────────────
    print("\n[2/2] Running detection on screenshots and images ...")
    detection_rows: list[dict] = []

    for idx, row in scraped_df.iterrows():
        url        = row.get("url", "unknown")
        screenshot = row.get("screenshot")
        images     = row.get("images", [])   # list of local image paths

        # Collect all image paths to check for this URL
        all_paths: list[str] = []
        if isinstance(screenshot, str) and os.path.exists(screenshot):
            all_paths.append(screenshot)
        if isinstance(images, list):
            all_paths.extend([p for p in images if isinstance(p, str) and os.path.exists(p)])

        if not all_paths:
            detection_rows.append({
                "url":          url,
                "dl_score":     0.0,
                "tm_score":     0.0,
                "final_score":  0.0,
                "logo_detected": False,
                "risk_level":   "LOW",
                "best_image":   None,
            })
            continue

        # Run detection on each image; keep the highest-scoring result
        best = {"final_score": -1.0}
        best_path = None
        for img_path in all_paths:
            det = detect_logo_in_image(img_path, dl_model, device, templates)
            if det["final_score"] > best["final_score"]:
                best      = det
                best_path = img_path

        risk = classify_risk(best["final_score"])

        print(f"  [{idx+1}] {url[:60]:<60}  "
              f"Score: {best['final_score']:.2f}  Risk: {risk}")

        detection_rows.append({
            "url":           url,
            "dl_score":      best["dl_score"],
            "tm_score":      best["tm_score"],
            "final_score":   best["final_score"],
            "logo_detected": best["logo_detected"],
            "risk_level":    risk,
            "best_image":    best_path,
        })

    # ── Save results ───────────────────────────────────────────────────────
    det_df = pd.DataFrame(detection_rows)

    out_path = os.path.join(config.OUTPUT_DIR, "detection_results.csv")
    det_df.to_csv(out_path, index=False)
    print(f"\n✓ Detection complete. Results saved to: {out_path}")

    flagged = det_df["logo_detected"].sum()
    print(f"  Logo detected on: {flagged}/{len(det_df)} websites")

    return det_df


# ─────────────────────────────────────────────────────────────────────────────
# Entry point — can run standalone for quick testing
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load scraping results from Module 2
    csv_path = os.path.join(config.OUTPUT_DIR, "scraped_results.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        # For testing: build a minimal DataFrame with dummy paths
        print("No scraped_results.csv found. Running training demo...")

        # Attempt training (will fail if logo_samples/ is empty)
        try:
            train_model()
        except RuntimeError as e:
            print(f"Training skipped: {e}")

        df = pd.DataFrame({
            "url":        ["https://example.com"],
            "screenshot": [None],
            "images":     [[]],
        })

    det_df = run_detection(df)
    print("\nDetection results:")
    print(det_df.to_string(index=False))
