import json
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from PIL import Image
from sklearn.metrics import confusion_matrix


# ==========================================
# 1. Geometric Feature Generator & Model Definitions (10 Classes: Lines, Cross, X, L-Shapes)
# ==========================================

class GeometricFeatureGenerator:
    """Layer 1: Generation and large-scale augmentation for refined geometric concepts (7x7)."""

    def __init__(self, image_size=(7, 7)):
        self.size = image_size
        # 10 classes focused on lines, cross, x-shape, and 4 directional L-shapes
        self.labels = [
            "vertical_line",    # 00: Vertical line
            "horizontal_line",  # 01: Horizontal line
            "diag_up",          # 02: Diagonal up-right
            "diag_down",        # 03: Diagonal down-right
            "cross",            # 04: Cross (+)
            "x_shape",          # 05: X-shape (×)
            "l_shape_ul",       # 06: L-shape (Top-Left open)
            "l_shape_ur",       # 07: L-shape (Top-Right open)
            "l_shape_dr",       # 08: L-shape (Bottom-Right open)
            "l_shape_dl"        # 09: L-shape (Bottom-Left open)
        ]
        self.label_to_id = {name: i for i, name in enumerate(self.labels)}

    def create_blank_image(self):
        return np.zeros((self.size[0], self.size[1], 1), dtype=np.float32)

    def _augment_image(self, base_img):
        augmented = [base_img.copy()]
        h, w = self.size

        # 1. Translation (Fine-grained shifts)
        for dx in [-2, -1, 1, 2]:
            M = np.float32([[1, 0, dx], [0, 1, 0]])
            augmented.append(cv2_warp(base_img, M, (w, h)))
        for dy in [-2, -1, 1, 2]:
            M = np.float32([[1, 0, 0], [0, 1, dy]])
            augmented.append(cv2_warp(base_img, M, (w, h)))

        # 2. Rotation (Mild range to preserve 7x7 primitive structure)
        center = (w // 2, h // 2)
        for angle in range(-12, 13, 3):
            if angle == 0:
                continue
            M = cv2.getRotationMatrix2D(center, float(angle), 1.0)
            rotated = cv2.warpAffine(base_img, M, (w, h))
            augmented.append(np.expand_dims(rotated, axis=-1) if rotated.ndim == 2 else rotated)

        # 3. Thickness variation (Dilation / Erosion)
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(base_img, kernel, iterations=1)
        eroded = cv2.erode(base_img, kernel, iterations=1)
        augmented.append(np.expand_dims(dilated, axis=-1) if dilated.ndim == 2 else dilated)
        augmented.append(np.expand_dims(eroded, axis=-1) if eroded.ndim == 2 else eroded)

        # 4. Noise injection
        for _ in range(5):
            noisy = base_img.copy()
            noise = np.random.normal(0, 0.1, size=(h, w, 1))
            noisy = np.clip(noisy + noise, 0.0, 1.0)
            augmented.append(noisy)

        return augmented

    def generate_base_samples(self):
        """Generates independent base patterns for each of the 10 classes."""
        base_samples = []

        def add_base(name, draw_func):
            img = self.create_blank_image()
            draw_func(img)
            base_samples.append((img, self.label_to_id[name]))

        # 00. vertical_line
        for x in [2, 3, 4]:
            add_base("vertical_line", lambda img, col=x: cv2.line(img, (col, 0), (col, 6), 1.0, 1))

        # 01. horizontal_line
        for y in [2, 3, 4]:
            add_base("horizontal_line", lambda img, row=y: cv2.line(img, (0, row), (6, row), 1.0, 1))

        # 02. diag_up
        for offset in [-1, 0, 1]:
            add_base("diag_up", lambda img, o=offset: cv2.line(img, (0+o, 6+o), (6+o, 0+o), 1.0, 1))

        # 03. diag_down
        for offset in [-1, 0, 1]:
            add_base("diag_down", lambda img, o=offset: cv2.line(img, (0+o, 0+o), (6+o, 6+o), 1.0, 1))

        # 04. cross
        add_base("cross", lambda img: (
            cv2.line(img, (0, 3), (6, 3), 1.0, 1),
            cv2.line(img, (3, 0), (3, 6), 1.0, 1)
        ))

        # 05. x_shape
        add_base("x_shape", lambda img: (
            cv2.line(img, (0, 0), (6, 6), 1.0, 1),
            cv2.line(img, (0, 6), (6, 0), 1.0, 1)
        ))

        # 06. l_shape_ul (Top-Left open)
        for c_pt, e1_pt, e2_pt in [
            ((1, 1), (1, 6), (6, 1)),
            ((1, 2), (1, 5), (5, 2)),
            ((2, 1), (2, 6), (6, 1))
        ]:
            add_base("l_shape_ul", lambda img, c=c_pt, p1=e1_pt, p2=e2_pt: (
                cv2.line(img, c, p1, 1.0, 1),
                cv2.line(img, c, p2, 1.0, 1)
            ))

        # 07. l_shape_ur (Top-Right open)
        for c_pt, e1_pt, e2_pt in [
            ((6, 1), (1, 1), (6, 6)),
            ((5, 2), (1, 2), (5, 5)),
            ((6, 2), (2, 2), (6, 6))
        ]:
            add_base("l_shape_ur", lambda img, c=c_pt, p1=e1_pt, p2=e2_pt: (
                cv2.line(img, c, p1, 1.0, 1),
                cv2.line(img, c, p2, 1.0, 1)
            ))

        # 08. l_shape_dr (Bottom-Right open)
        for c_pt, e1_pt, e2_pt in [
            ((6, 6), (1, 6), (6, 1)),
            ((5, 5), (1, 5), (5, 1)),
            ((6, 5), (2, 5), (6, 1))
        ]:
            add_base("l_shape_dr", lambda img, c=c_pt, p1=e1_pt, p2=e2_pt: (
                cv2.line(img, c, p1, 1.0, 1),
                cv2.line(img, c, p2, 1.0, 1)
            ))

        # 09. l_shape_dl (Bottom-Left open)
        for c_pt, e1_pt, e2_pt in [
            ((1, 6), (1, 1), (6, 6)),
            ((2, 5), (2, 1), (5, 5)),
            ((1, 5), (1, 2), (5, 5))
        ]:
            add_base("l_shape_dl", lambda img, c=c_pt, p1=e1_pt, p2=e2_pt: (
                cv2.line(img, c, p1, 1.0, 1),
                cv2.line(img, c, p2, 1.0, 1)
            ))

        return base_samples

    def generate_dataset_samples(self):
        base_samples = self.generate_base_samples()
        dataset = []
        for img, label_id in base_samples:
            for aug_img in self._augment_image(img):
                dataset.append((aug_img, label_id))
        return dataset

def cv2_warp(img, M, size):
    shifted = cv2.warpAffine(img, M, size)
    return np.expand_dims(shifted, axis=-1) if shifted.ndim == 2 else shifted


class GeometricFilter(nn.Module):
    """[Geometric Filter] Input: 7x7x1 -> Output: 16-dimensional vector space"""
    def __init__(self):
        super(GeometricFilter, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 3 * 3, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 16)  # 16-dim vector space
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ConceptDictionary(nn.Module):
    """[Concept Label Dictionary Model] Input: 16-dim -> Output: Geometric concept logits (10 classes)"""
    def __init__(self, num_classes=10):
        super(ConceptDictionary, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)


class FullGeometricConceptModel(nn.Module):
    """Integrated Model (Geometric Filter + Concept Dictionary)"""
    def __init__(self, num_classes=10):
        super(FullGeometricConceptModel, self).__init__()
        self.filter_model = GeometricFilter()
        self.dict_model = ConceptDictionary(num_classes)

    def forward(self, x):
        feat = self.filter_model(x)
        out = self.dict_model(feat)
        return out


class GeometricPipeline:
    def __init__(self, device="cpu", config=None):
        self.device = device
        self.config = config
        self.space_dir = config.space_dir if config and hasattr(config, "space_dir") else "space"
        os.makedirs(self.space_dir, exist_ok=True)
        self.ckpt_filter_path = os.path.join(self.space_dir, "checkpoint_geometric_filter.pth")
        self.ckpt_dict_path = os.path.join(self.space_dir, "checkpoint_concept_dictionary.pth")
        self.labels_json_path = os.path.join(self.space_dir, "concept_labels.json")
        self.label_images_dir = os.path.join(self.space_dir, "label_image")
        os.makedirs(self.label_images_dir, exist_ok=True)

    def train(self):
        print("\n🚀 [Geometric Train] Starting training with 10 Clean Geometric Concepts (50 epochs)...")

        generator = GeometricFeatureGenerator(image_size=(7, 7))
        base_samples = generator.generate_base_samples()
        
        # Save base reference images scaled up 5x
        scale_factor = 5
        saved_labels = set()
        for img, label_id in base_samples:
            label_name = generator.labels[label_id]
            if label_name in saved_labels:
                continue
            
            img_uint8 = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
            if img_uint8.ndim == 3 and img_uint8.shape[2] == 1:
                img_uint8 = img_uint8.squeeze(axis=2)
            
            img_pil = Image.fromarray(img_uint8)
            img_pil = img_pil.resize((img_pil.width * scale_factor, img_pil.height * scale_factor), Image.NEAREST)
            
            save_path = os.path.join(self.label_images_dir, f"{label_id:02d}_{label_name}.png")
            img_pil.save(save_path)
            saved_labels.add(label_name)
                
        print(f"📁 Saved {len(saved_labels)} centralized base concept reference images (5x enlarged) to: {self.label_images_dir}")

        raw_samples = generator.generate_dataset_samples()
        print(f"  - Total samples (augmented): {len(raw_samples)}")

        X_list, y_list = [], []
        for img, label_id in raw_samples:
            X_list.append(torch.tensor(img, dtype=torch.float32).permute(2, 0, 1))
            y_list.append(label_id)

        X = torch.stack(X_list)
        y = torch.tensor(y_list, dtype=torch.long)

        dataset = TensorDataset(X, y)
        train_size = int(0.85 * len(dataset))
        val_size = len(dataset) - train_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

        num_classes = len(generator.labels)
        full_model = FullGeometricConceptModel(num_classes=num_classes).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(full_model.parameters(), lr=0.001)
        
        epochs = 50
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        for epoch in range(epochs):
            full_model.train()
            total_loss, correct, total = 0, 0, 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = full_model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * inputs.size(0)
                _, preds = outputs.max(1)
                correct += preds.eq(targets).sum().item()
                total += targets.size(0)

            scheduler.step()
            train_acc = 100.0 * correct / total

            full_model.eval()
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = full_model(inputs)
                    _, preds = outputs.max(1)
                    val_correct += preds.eq(targets).sum().item()
                    val_total += targets.size(0)

            val_acc = 100.0 * val_correct / val_total
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/total:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        self.evaluate_concept_dictionary(full_model, val_loader, generator.labels)

        torch.save({"model_state_dict": full_model.filter_model.state_dict(), "labels": generator.labels}, self.ckpt_filter_path)
        torch.save({"model_state_dict": full_model.dict_model.state_dict()}, self.ckpt_dict_path)
        
        with open(self.labels_json_path, "w", encoding="utf-8") as f:
            json.dump({"labels": generator.labels, "label_to_id": generator.label_to_id}, f, ensure_ascii=False, indent=4)

        print(f"✅ Geometric filter saved to: {self.ckpt_filter_path}")
        print(f"✅ Concept dictionary model saved to: {self.ckpt_dict_path}")

    def evaluate_concept_dictionary(self, model, val_loader, labels):
        """[Concept Dictionary Test] Evaluate per-class accuracy and safe confusion matrix."""
        print("\n=== Geometric Concept Dictionary Test ===")
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                _, preds = outputs.max(1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        class_acc = {}
        for i, label in enumerate(labels):
            mask = (all_targets == i)
            if mask.sum() > 0:
                acc = (all_preds[mask] == i).sum() / mask.sum() * 100.0
                class_acc[label] = acc
            else:
                class_acc[label] = 0.0

        print("\n--- Per-Class Accuracy (acc[label]) ---")
        sorted_acc = sorted(class_acc.items(), key=lambda x: x[1])
        for label, acc in sorted_acc:
            status = "🔴 WEAK" if acc < 70.0 else "🟢 GOOD" if acc > 90.0 else "🟡 FAIR"
            print(f"  {label:<20} : {acc:6.2f}% {status}")

        print("\n--- Confusion Matrix & Misclassification Analysis ---")
        cm = confusion_matrix(all_targets, all_preds, labels=range(len(labels)))
        confusions = []
        for i in range(len(labels)):
            for j in range(len(labels)):
                if i != j and cm[i, j] > 0:
                    confusions.append((cm[i, j], labels[i], labels[j]))
        
        confusions.sort(key=lambda x: x[0], reverse=True)
        if confusions:
            for count, true_lbl, pred_lbl in confusions[:10]:
                print(f"  {true_lbl}  ==>  misclassified as  {pred_lbl}  ({count} times)")
        else:
            print("  No major misclassifications found in validation set!")