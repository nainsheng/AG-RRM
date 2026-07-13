# ============================================
# AGRRM Evaluation Dataset
# ============================================

import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset


class FineGrainedDataset(Dataset):
    """
    Evaluation dataset: uses JSON only for face cropping (bbox).
    No part masks are loaded, making it faster and JSON-light.

    Args:
        img_dir: Directory containing images
        angle_dir: Directory containing angle .txt files
        json_dir: Directory containing JSON files (only 'bbox' is used)
        transform: Image preprocessing pipeline
        input_size: Target size for resizing after cropping
    """

    def __init__(self, img_dir, angle_dir, json_dir, transform=None, input_size=448):
        self.img_dir = img_dir
        self.angle_dir = angle_dir
        self.json_dir = json_dir
        self.transform = transform
        self.input_size = input_size

        self.files = [f for f in os.listdir(img_dir)
                      if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        self.files.sort()

        # Extract identity IDs
        unique_ids = sorted(list(set([f.split('_')[0] for f in self.files])))
        self.id_to_label = {pid: i for i, pid in enumerate(unique_ids)}

        # Pre-load labels and angles
        self.labels = []
        self.angles = []

        for fname in self.files:
            pid = fname.split('_')[0]
            self.labels.append(self.id_to_label[pid])

            # Read yaw angle
            base_name = os.path.splitext(fname)[0]
            angle_path = os.path.join(angle_dir, base_name + ".txt")
            yaw = 0.0
            if os.path.exists(angle_path):
                try:
                    with open(angle_path, 'r') as f:
                        for line in f:
                            if "Yaw" in line:
                                yaw = float(line.split(":")[1].strip()) / 90.0
                                break
                except:
                    pass
            self.angles.append(yaw)

        print(f"Dataset loaded: {len(self.files)} images, {len(unique_ids)} identities")

    def __len__(self):
        return len(self.files)

    def _get_face_box(self, json_path):
        """Read only the face bounding box from JSON, ignore part annotations"""
        if not os.path.exists(json_path):
            return None

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                shapes = data.get('shapes', [])
                for shape in shapes:
                    label = shape['label'].lower().strip()
                    if label == 'bbox':
                        pts = shape['points']
                        x_min, y_min = pts[0]
                        x_max, y_max = pts[1]
                        return [x_min, y_min, x_max, y_max]
        except:
            pass
        return None

    def __getitem__(self, idx):
        fname = self.files[idx]
        base_name = os.path.splitext(fname)[0]

        try:
            img = Image.open(os.path.join(self.img_dir, fname)).convert('RGB')
        except:
            img = Image.new('RGB', (self.input_size, self.input_size), color='black')

        # Try to crop face using bbox from JSON (if available)
        json_path = os.path.join(self.json_dir, base_name + ".json")
        face_box = self._get_face_box(json_path)

        if face_box is not None:
            try:
                x1, y1, x2, y2 = face_box
                margin = 20
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(img.width, x2 + margin)
                y2 = min(img.height, y2 + margin)
                img = img.crop((x1, y1, x2, y2))
            except:
                pass

        if self.transform:
            img = self.transform(img)

        return {
            'img': img,
            'label': self.labels[idx],
            'angle': torch.tensor(self.angles[idx], dtype=torch.float32),
            'fname': fname
        }