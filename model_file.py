#!/usr/bin/env python3
"""
app.py

Single-file training + evaluation + similarity search tool for your
skin lesion classifier based on EfficientNet-B2.

Expected dataset layout (Option A):
dataset/
  train/
    benign/
    malignant/
  val/
    benign/
    malignant/
  test/
    benign/
    malignant/

Commands:
  python app.py train                 # train model and build feature DB
  python app.py build-db              # build feature DB from dataset/train
  python app.py predict --image IMG   # predict + neighbor voting for IMG
"""

import os
import random
import shutil
import time
from glob import glob
from collections import defaultdict
import argparse

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# --------------------
# SETTINGS
# --------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# You already created train/val/test folders, so disable auto-split
AUTO_CREATE_SPLIT = False
EARLY_STOPPING = True
PATIENCE = 3
MIN_DELTA = 1e-4

# Feature DB path
FEATURE_DB_PATH = "results/features.npz"
TOPK = 10

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------
# UTIL: parse patient id from filename
# --------------------


def parse_patient_from_filename(filename):
    base = os.path.splitext(os.path.basename(filename))[0]
    parts = base.split("_")
    if len(parts) >= 3 and parts[0].upper().startswith("PAT"):
        if parts[0].upper() == "PAT":
            patient = f"PAT_{parts[1]}"
        else:
            if "_" in parts[0]:
                patient = parts[0]
            else:
                patient = parts[0] if parts[0].upper().startswith(
                    "PAT_") else "PAT_" + parts[1]
        return patient
    return None

# --------------------
# MODEL BUILDERS / ENCODER
# --------------------


def build_efficientnet_b2(num_classes=2, weights_enum=None):
    """Build EfficientNet-B2 and replace classifier head robustly."""
    try:
        if weights_enum is not None:
            model = models.efficientnet_b2(weights=weights_enum)
        else:
            model = models.efficientnet_b2(weights=None)
    except Exception:
        model = models.efficientnet_b2(weights=None)

    # Replace classifier head in a robust way
    try:
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    except Exception:
        try:
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
        except Exception:
            if hasattr(model, "fc"):
                in_features = model.fc.in_features
                model.fc = nn.Linear(in_features, num_classes)
            else:
                raise RuntimeError(
                    "Unable to replace EfficientNet classifier head for this torchvision version.")
    return model


def make_feature_encoder(model):
    """
    Return an encoder (nn.Module) that maps input images -> embedding vector.
    Using model.features + AdaptiveAvgPool2d + Flatten produces a vector.
    """
    encoder = nn.Sequential(
        model.features,
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten()
    )
    return encoder

# --------------------
# EVALUATION (robust)
# --------------------


def evaluate_model(model, data_loader, device, save_dir="results", prefix="eval"):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    # detect which index is malignant from dataset classes
    positive_index = 1
    try:
        cls_to_idx = data_loader.dataset.class_to_idx
        for k, v in cls_to_idx.items():
            if "mal" in k.lower():
                positive_index = v
                break
        print("Class to idx mapping (evaluation):", cls_to_idx,
              " -> positive_index:", positive_index)
    except Exception:
        pass

    all_labels, all_predictions, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, positive_index].cpu().numpy())

    cm = confusion_matrix(all_labels, all_predictions)
    target_names = list(data_loader.dataset.classes) if hasattr(
        data_loader.dataset, "classes") else ["Benign", "Malignant"]
    report = classification_report(
        all_labels, all_predictions, target_names=target_names, digits=4)

    try:
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            sensitivity = specificity = 0.0
    except Exception:
        sensitivity = specificity = 0.0

    try:
        fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
    except Exception:
        roc_auc = None

    # Save textual summary
    with open(os.path.join(save_dir, f"{prefix}_summary.txt"), "w") as f:
        f.write(f"Confusion Matrix:\n{cm}\n\n")
        f.write(f"Classification Report:\n{report}\n")
        f.write(f"Sensitivity: {sensitivity:.4f}\n")
        f.write(f"Specificity: {specificity:.4f}\n")
        f.write(f"AUC: {roc_auc}\n")

    # ROC plot
    if roc_auc is not None:
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], "--")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title(f"ROC Curve ({prefix})")
        plt.legend()
        plt.savefig(os.path.join(save_dir, f"{prefix}_roc.png"), dpi=300)
        plt.close()

    # confusion matrix plot
    plt.figure(figsize=(4, 3))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix ({prefix})")
    plt.colorbar()
    tick = np.arange(len(target_names))
    plt.xticks(tick, target_names)
    plt.yticks(tick, target_names)
    thresh = cm.max() / 2 if cm.size > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]), ha="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_confusion.png"), dpi=300)
    plt.close()

    return {
        "cm": cm,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "auc": roc_auc,
        "report": report
    }

# --------------------
# FEATURE DB BUILDING
# --------------------


def build_feature_db(model, dataset_dir="dataset/train", device="cpu", transform=None, out_path=FEATURE_DB_PATH, batch_size=16):
    """
    Build features for all images under dataset_dir using ImageFolder structure.
    Saves npz with keys: features (N,D), labels (N,), paths (N,), classes (array)
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    dataset = datasets.ImageFolder(dataset_dir, transform=transform)
    if len(dataset) == 0:
        raise RuntimeError(
            f"No images found in {dataset_dir} - cannot build feature DB.")

    encoder = make_feature_encoder(model).to(device)
    encoder.eval()

    features = []
    labels = []
    paths = []

    # dataset.samples is list[(path, class_idx)] in the same order as dataset
    samples = dataset.samples
    n = len(samples)
    i = 0
    while i < n:
        batch_samples = samples[i:i+batch_size]
        imgs = []
        labs = []
        for p, lab in batch_samples:
            img = Image.open(p).convert("RGB")
            img = transform(img)  # tensor
            imgs.append(img)
            labs.append(lab)
            paths.append(p)
        x = torch.stack(imgs).to(device)
        with torch.no_grad():
            vecs = encoder(x).cpu().numpy()  # (B, D)
        features.append(vecs)
        labels.extend(labs)
        i += batch_size

    features = np.vstack(features).astype(np.float32)  # (N, D)
    labels = np.array(labels, dtype=np.int32)
    paths = np.array(paths, dtype=object)
    # normalize rows (L2) to unit length for cosine similarity
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    features = features / norms

    np.savez_compressed(out_path, features=features, labels=labels,
                        paths=paths, classes=np.array(dataset.classes))
    print("Feature DB saved to", out_path, "features shape:", features.shape)
    return out_path

# --------------------
# SIMILARITY / SEARCH
# --------------------


def cosine_similarity_vector(query_vec, db_feats):
    # both query_vec (D,), db_feats (N,D) assumed L2-normalized -> dot product = cosine
    return np.dot(db_feats, query_vec)  # returns (N,)


def find_similar_images(query_path, model, feature_db_path=FEATURE_DB_PATH, k=TOPK, device="cpu", transform=None, montage_out="results/top10.png"):
    """
    Return top-k similar images from feature DB (build DB if missing).
    Returns dict with topk_paths, topk_scores, topk_labels, neighbor_fraction, classes.
    """
    os.makedirs(os.path.dirname(montage_out) or ".", exist_ok=True)

    # ensure DB exists
    if not os.path.exists(feature_db_path):
        print("Feature DB not found; building from dataset/train...")
        weights_enum = None
        try:
            weights_enum = models.EfficientNet_B2_Weights.IMAGENET1K_V1
        except Exception:
            weights_enum = None
        tmp_model = build_efficientnet_b2(
            num_classes=2, weights_enum=weights_enum)
        build_feature_db(tmp_model, dataset_dir="dataset/train",
                         device=device, transform=transform, out_path=feature_db_path)

    data = np.load(feature_db_path, allow_pickle=True)
    db_feats = data["features"]   # (N, D)
    db_labels = data["labels"]    # (N,)
    db_paths = data["paths"]      # (N,)
    classes = list(data["classes"])

    # build encoder from provided model
    encoder = make_feature_encoder(model).to(device)
    encoder.eval()

    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    img = Image.open(query_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        qvec = encoder(x).cpu().numpy().reshape(-1)
    qvec = qvec / (np.linalg.norm(qvec) + 1e-10)

    sims = cosine_similarity_vector(qvec, db_feats)  # (N,)
    idxs = np.argsort(-sims)[:k]
    topk_paths = db_paths[idxs].tolist()
    topk_scores = sims[idxs].tolist()
    topk_labels = db_labels[idxs].tolist()

    # detect which class index is malignant
    mal_idx_in_classes = None
    for i, cname in enumerate(classes):
        if "mal" in str(cname).lower() or "cancer" in str(cname).lower():
            mal_idx_in_classes = i
            break
    if mal_idx_in_classes is None and len(classes) >= 2:
        mal_idx_in_classes = 1

    neighbor_malignant = sum(
        1 for lab in topk_labels if lab == mal_idx_in_classes)
    neighbor_fraction = neighbor_malignant / max(1, len(topk_labels))

    # Save montage (simple grid)
    try:
        imgs = [Image.open(p).convert("RGB").resize((224, 224))
                for p in topk_paths]
        cols = min(5, len(imgs))
        rows = (len(imgs)+cols-1)//cols
        w, h = imgs[0].size
        montage = Image.new("RGB", (cols*w, rows*h), (255, 255, 255))
        for i, im in enumerate(imgs):
            x = (i % cols) * w
            y = (i // cols) * h
            montage.paste(im, (x, y))
        montage.save(montage_out)
    except Exception as e:
        print("Could not save montage:", e)

    return {
        "topk_paths": topk_paths,
        "topk_scores": topk_scores,
        "topk_labels": topk_labels,
        "neighbor_fraction": neighbor_fraction,
        "classes": classes
    }

# --------------------
# PREDICTION: model + similarity combined
# --------------------


def predict_image_with_similarity(image_path, model_path="results/best_model.pth", device=DEVICE, weight_model=0.5, weight_neighbors=0.5):
    """
    Returns:
      model_prediction, model_probs, neighbor_fraction, neighbor_pct, combined_pct, risk, topk_paths
    """
    # prepare model
    weights_enum = None
    try:
        weights_enum = models.EfficientNet_B2_Weights.IMAGENET1K_V1
    except Exception:
        weights_enum = None
    model = build_efficientnet_b2(num_classes=2, weights_enum=weights_enum)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
    model_benign = float(probs[0]*100)
    model_malignant = float(probs[1]*100)
    model_pred = "Malignant" if model_malignant > model_benign else "Benign"

    sim_result = find_similar_images(image_path, model, feature_db_path=FEATURE_DB_PATH,
                                     k=TOPK, device=device, transform=transform, montage_out="results/top10.png")
    neighbor_fraction = sim_result["neighbor_fraction"]
    neighbor_pct = neighbor_fraction * 100.0

    combined = weight_model * model_malignant + weight_neighbors * neighbor_pct
    combined = float(round(combined, 2))

    if combined >= 70:
        risk = "HIGH RISK"
    elif combined >= 40:
        risk = "MEDIUM RISK"
    else:
        risk = "LOW RISK"

    return {
        "model_prediction": model_pred,
        "model_benign_pct": round(model_benign, 2),
        "model_malignant_pct": round(model_malignant, 2),
        "neighbor_malignant_fraction": round(neighbor_fraction, 3),
        "neighbor_malignant_pct": round(neighbor_pct, 2),
        "combined_malignant_pct": combined,
        "risk": risk,
        "topk_paths": sim_result.get("topk_paths", []),
        "topk_scores": sim_result.get("topk_scores", []),
        "classes": sim_result.get("classes", [])
    }

# --------------------
# TRAINING pipeline
# --------------------


def train_model(save_dir="results"):
    device = DEVICE
    print("Using device:", device)
    os.makedirs(save_dir, exist_ok=True)

    if AUTO_CREATE_SPLIT:
        # we left patient-level split code out because you already created splits
        pass

    train_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.15, 0.15, 0.15, 0.02),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_set = datasets.ImageFolder("dataset/train", transform=train_tf)
    val_set = datasets.ImageFolder("dataset/val", transform=val_tf)
    test_set = datasets.ImageFolder("dataset/test", transform=val_tf)

    print("Class mapping (train):", train_set.class_to_idx)

    train_loader = DataLoader(train_set, batch_size=8,
                              shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=8,
                            shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=8,
                             shuffle=False, num_workers=2)

    print("Dataset sizes ->", len(train_set), len(val_set), len(test_set))

    # build model
    weights_enum = None
    try:
        weights_enum = models.EfficientNet_B2_Weights.IMAGENET1K_V1
    except Exception:
        weights_enum = None

    model = build_efficientnet_b2(num_classes=2, weights_enum=weights_enum)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss()

    # ---- LR Scheduler (fixed version, NO 'verbose' argument) ----
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    epochs = 12
    best_val_acc = 0
    best_val_loss = float("inf")
    es_counter = 0

    train_losses, val_losses, val_accs = [], [], []

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # -------- TRAIN --------
        model.train()
        running = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            running += loss.item() * images.size(0)

        train_loss = running / len(train_loader.dataset)
        train_losses.append(train_loss)

        # -------- VALIDATION --------
        model.eval()
        vloss = 0.0
        correct, total = 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                out = model(images)
                loss = criterion(out, labels)
                vloss += loss.item() * images.size(0)

                preds = torch.argmax(F.softmax(out, dim=1), 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss = vloss / len(val_loader.dataset)
        val_acc = correct / total * 100
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # ---- Scheduler Step ----
        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr != old_lr:
            print(f"[LR Scheduler] Learning rate reduced: {old_lr} → {new_lr}")

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train {train_loss:.4f} | Val {val_loss:.4f} | "
            f"Acc {val_acc:.2f}% | {time.time() - t0:.1f}s"
        )

        # ---- Save Best Model ----
        if val_acc > best_val_acc:
            torch.save(model.state_dict(), os.path.join(
                save_dir, "best_model.pth"))
            best_val_acc = val_acc
            print("🔥 New best model saved.")

        # ---- Early Stopping ----
        if val_loss + MIN_DELTA < best_val_loss:
            best_val_loss = val_loss
            es_counter = 0
        else:
            es_counter += 1
            if es_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    # -------- Plot Learning Curves --------
    ep_range = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(ep_range, train_losses, label="Train")
    plt.plot(ep_range, val_losses, label="Val")
    plt.legend()
    plt.title("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(ep_range, val_accs, label="Val Acc")
    plt.legend()
    plt.title("Accuracy")

    plt.savefig(os.path.join(save_dir, "learning_curves.png"), dpi=300)
    plt.close()

    # -------- Evaluate Best Model on Test Set --------
    print("Evaluating BEST model on TEST set...")
    model.load_state_dict(torch.load(os.path.join(
        save_dir, "best_model.pth"), map_location=device))
    test_metrics = evaluate_model(model, test_loader, device, prefix="test")

    with open(os.path.join(save_dir, "test_results.txt"), "w") as f:
        f.write(f"Best Validation Accuracy: {best_val_acc:.2f}\n")
        f.write(f"Sensitivity: {test_metrics['sensitivity']:.4f}\n")
        f.write(f"Specificity: {test_metrics['specificity']:.4f}\n")
        f.write(f"AUC: {test_metrics['auc']}\n")
        f.write(test_metrics["report"])

    # -------- Build Feature DB (if missing) --------
    if not os.path.exists(FEATURE_DB_PATH):
        print("Building feature DB from trained model...")
        build_feature_db(model, dataset_dir="dataset/train", device=device)

    return save_dir + "/best_model.pth"


# --------------------
# MAIN
# --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "build-db", "predict"],
                        help="train, build-db, or predict")
    parser.add_argument("--image", type=str, default=None,
                        help="Image path for prediction mode")
    parser.add_argument("--model", type=str, default="results/best_model.pth",
                        help="Path to trained model")
    parser.add_argument("--weight_model", type=float, default=0.5,
                        help="Weight of model's prediction in final score")
    parser.add_argument("--weight_neighbors", type=float, default=0.5,
                        help="Weight of neighbor votes in final score")
    args = parser.parse_args()

    if args.mode == "train":
        train_model()

    elif args.mode == "build-db":
        print("Building feature DB...")
        # Load untrained EfficientNet for encoder
        try:
            weights_enum = models.EfficientNet_B2_Weights.IMAGENET1K_V1
        except Exception:
            weights_enum = None
        model = build_efficientnet_b2(num_classes=2, weights_enum=weights_enum)
        build_feature_db(model, dataset_dir="dataset/train", device=DEVICE)

    elif args.mode == "predict":
        if not args.image:
            print("Please provide --image path for prediction mode.")
            return

        res = predict_image_with_similarity(
            args.image,
            model_path=args.model,
            device=DEVICE,
            weight_model=args.weight_model,
            weight_neighbors=args.weight_neighbors
        )

        print("\n=== Prediction Summary ===")
        print("Model prediction:", res["model_prediction"])
        print(
            f"Model malignant %: {res['model_malignant_pct']} | "
            f"Neighbor malignant %: {res['neighbor_malignant_pct']} | "
            f"Combined: {res['combined_malignant_pct']}%"
        )
        print("Risk level:", res["risk"])
        print("Top matched images saved to results/top10.png")
        print("Top paths:", res["topk_paths"][:5])

    else:
        print("Unknown mode.")


if __name__ == "__main__":
    main()

