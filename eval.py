# ============================ AGRRM Evaluation ============================
import os
import sys
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import gc
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================== Dataset Paths ======================
BASE_DIR = r"dataset"
GALLERY_DIR = f"{BASE_DIR}/gallery/image"
PROBE_DIR = f"{BASE_DIR}/probe/image"

SAVE_DIR = r'Result'
os.makedirs(SAVE_DIR, exist_ok=True)

from EfficientNetB3_AGRRM_ReID import EfficientNetB3_AGRRM_ReID
from FineGrainedDataset import FineGrainedDataset


# ============================================
# Evaluation Function
# ============================================
@torch.no_grad()
def evaluate_agrrm(model, gallery_loader, probe_loader, device, model_name="AGRRM"):
    """
    Evaluate AGRRM model, return Top-1/5/10 accuracy and mAP
    """
    model.eval()

    def extract_batch(loader, desc):
        feats, labels = [], []
        for batch in tqdm(loader, desc=desc):
            imgs = batch['img'].to(device)
            angles = batch['angle'].to(device).unsqueeze(1)
            feat = model.extract_features(imgs, angles)
            feats.append(feat.cpu())
            labels.append(batch['label'])
        return torch.cat(feats, dim=0), torch.cat(labels)

    print("Extracting Gallery features...")
    gallery_feats, gallery_labels = extract_batch(gallery_loader, "Gallery")

    print("Extracting Probe features...")
    probe_feats, probe_labels = extract_batch(probe_loader, "Probe")

    # L2 normalization
    gallery_feats = F.normalize(gallery_feats, p=2, dim=1)
    probe_feats = F.normalize(probe_feats, p=2, dim=1)

    # Cosine similarity matrix
    sim = torch.mm(probe_feats, gallery_feats.T).numpy()

    gl = gallery_labels.numpy()
    pl = probe_labels.numpy()

    # Compute Top-k
    results = {}
    for k in [1, 5, 10]:
        correct = 0
        for i in range(len(pl)):
            topk = np.argsort(sim[i])[::-1][:k]
            if pl[i] in gl[topk]:
                correct += 1
        results[f'top{k}'] = correct / len(pl)

    # Compute mAP
    print("Computing mAP...")
    ap_list = []
    for i in range(len(pl)):
        sorted_indices = np.argsort(sim[i])[::-1]
        matches = (gl[sorted_indices] == pl[i]).astype(np.float32)
        cumsum = np.cumsum(matches)
        precision = cumsum / (np.arange(len(matches)) + 1)
        ap = np.sum(precision * matches) / max(np.sum(matches), 1)
        ap_list.append(ap)
    results['mAP'] = np.mean(ap_list)

    return results, sim, gl, pl


# ============================================
# Save Results
# ============================================
def save_results(model_name, results, save_dir):
    save_path = os.path.join(save_dir, f"{model_name}_results.txt")
    with open(save_path, 'w') as f:
        f.write(f"{model_name} Evaluation Results\n")
        f.write("=" * 60 + "\n")
        for k in [1, 5, 10]:
            f.write(f"Top-{k}: {results[f'top{k}']:.4f} ({results[f'top{k}']*100:.2f}%)\n")
        f.write(f"mAP:    {results['mAP']:.4f} ({results['mAP']*100:.2f}%)\n")
    print(f"Results saved: {save_path}")


# ============================================
# Main Function
# ============================================
def main():
    print("=" * 60)
    print("AG-RRM Evaluation")
    print("=" * 60)

    # Data preprocessing
    tf_val = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load datasets
    gallery_dataset = FineGrainedDataset(
        img_dir=GALLERY_DIR,
        angle_dir=f"{BASE_DIR}/gallery/angle",
        json_dir=f"{BASE_DIR}/gallery/json",
        transform=tf_val,
        input_size=448
    )

    probe_dataset = FineGrainedDataset(
        img_dir=PROBE_DIR,
        angle_dir=f"{BASE_DIR}/probe/angle",
        json_dir=f"{BASE_DIR}/probe/json",
        transform=tf_val,
        input_size=448
    )

    print(f"Gallery: {len(gallery_dataset)} images | Probe: {len(probe_dataset)} images")

    gallery_loader = torch.utils.data.DataLoader(
        gallery_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )
    probe_loader = torch.utils.data.DataLoader(
        probe_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )

    # Load model
    CKPT = r'model/best_model.pth'

    model = EfficientNetB3_AGRRM_ReID(num_classes=204)
    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt, strict=False)
    model.to(DEVICE).eval()

    print(f"Model loaded: {CKPT}")

    # Evaluate
    results, sim, gl, pl = evaluate_agrrm(
        model, gallery_loader, probe_loader, DEVICE, model_name="AGRRM"
    )

    # Print results
    print("\n" + "=" * 60)
    print("AGRRM Evaluation Results")
    print("=" * 60)
    for k in [1, 5, 10]:
        print(f"Top-{k}: {results[f'top{k}']:.4f} ({results[f'top{k}']*100:.2f}%)")
    print(f"mAP:    {results['mAP']:.4f} ({results['mAP']*100:.2f}%)")

    # Save results
    save_results("AGRRM", results, SAVE_DIR)

    print(f"\nEvaluation complete! Results saved in: {SAVE_DIR}")


if __name__ == "__main__":
    main()