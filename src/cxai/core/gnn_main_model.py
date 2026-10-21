import json
import logging
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from PIL import Image

from .mnist_dataloader import MNISTDataLoader
from .gnn_preset_model import GeometricFilter, ConceptDictionary
from .gnn_main_model_helper import TrainDataEvaluator

logger = logging.getLogger(__name__)


class GeometricExtractionLayer(nn.Module):
    """
    Custom layer fully optimized for GPU (Zero Python loops in NMS/extraction, 
    batch-wise torch.where, and pre-allocated padding template).
    Returns tensor of shape: (B, max_boxes=20, 20) -> [x1, y1, x2, y2, concept_vector(16dim)]
    """

    def __init__(self, filter_model_path, dict_model_path, labels_json_path, default_labels, output_dir="space/output", filter_size=7, stride=1, max_boxes=20):
        super(GeometricExtractionLayer, self).__init__()
        self.filter_size = filter_size
        self.stride = stride
        self.max_boxes = max_boxes
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        if os.path.exists(labels_json_path):
            try:
                with open(labels_json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.labels = data.get("labels", default_labels)
            except Exception:
                self.labels = default_labels
        else:
            self.labels = default_labels
        self.num_classes = len(self.labels)

        self.filter_model = GeometricFilter()
        if os.path.exists(filter_model_path):
            ckpt = torch.load(filter_model_path, map_location="cpu", weights_only=False)
            self.filter_model.load_state_dict(ckpt["model_state_dict"])
        self.filter_model.eval()
        for param in self.filter_model.parameters():
            param.requires_grad = False

        self.dict_model = ConceptDictionary(num_classes=self.num_classes)
        if os.path.exists(dict_model_path):
            ckpt = torch.load(dict_model_path, map_location="cpu", weights_only=False)
            self.dict_model.load_state_dict(ckpt["model_state_dict"])
        self.dict_model.eval()
        for param in self.dict_model.parameters():
            param.requires_grad = False

        self.maxpool = nn.MaxPool2d(
            kernel_size=self.filter_size,
            stride=1,
            padding=self.filter_size // 2
        )

        self.register_buffer(
            "padding_template",
            torch.full((self.max_boxes, 20), -1.0, dtype=torch.float32)
        )

        self.log_counter = 0
        self.current_epoch = 0  # Tracks the current epoch index for conditional logging
        self.last_extracted_records = None

    def forward(self, x):
        device = x.device
        N, c, h, w = x.shape
        fs = self.filter_size
        
        h_grid = (h - fs) // self.stride + 1
        w_grid = (w - fs) // self.stride + 1

        patches_unfolded = nn.functional.unfold(x, kernel_size=fs, stride=self.stride)
        L = patches_unfolded.shape[-1]
        patches_flat = patches_unfolded.permute(0, 2, 1).reshape(N * L, c, fs, fs)

        # 1. Extract 16-dimensional features via the geometric filter (batched GPU operation)
        feat_16d_flat = self.filter_model(patches_flat) # (N*L, 16)
        spatial_feats_16 = feat_16d_flat.reshape(N, h_grid, w_grid, 16) # (N, h_grid, w_grid, 16)

        # 2. Perform NMS and confidence thresholding (fully vectorized)
        with torch.no_grad():
            logits = self.dict_model(feat_16d_flat)
            probs = torch.softmax(logits, dim=1)
            max_probs, _ = torch.max(probs, dim=1) # (N*L,)
            
            score_map = max_probs.reshape(N, 1, h_grid, w_grid)
            local_max = self.maxpool(score_map)
            
            is_local_max = (score_map == local_max)
            is_confident = (score_map >= 0.8)
            peak_mask = (is_local_max & is_confident).squeeze(1) # (N, h_grid, w_grid)

            batch_idx, y_indices, x_indices = torch.where(peak_mask)

            output_tensor = self.padding_template.unsqueeze(0).expand(N, -1, -1).clone()

            if batch_idx.numel() > 0:
                x1 = x_indices.float() * self.stride
                y1 = y_indices.float() * self.stride
                x2 = x1 + fs
                y2 = y1 + fs

                coords = torch.stack([x1, y1, x2, y2], dim=-1) # (total_peaks, 4)
                concept_vecs = spatial_feats_16[batch_idx, y_indices, x_indices] # (total_peaks, 16)
                all_records = torch.cat([coords, concept_vecs], dim=-1) # (total_peaks, 20)

                box_counts = torch.bincount(batch_idx, minlength=N)
                start_indices = torch.cat([torch.tensor([0], device=device), box_counts[:-1].cumsum(0)])
                arange_indices = torch.arange(batch_idx.numel(), device=device)
                within_batch_indices = arange_indices - start_indices[batch_idx]

                valid_mask = within_batch_indices < self.max_boxes
                if valid_mask.any():
                    b_valid = batch_idx[valid_mask]
                    w_valid = within_batch_indices[valid_mask]
                    rec_valid = all_records[valid_mask]
                    
                    output_tensor[b_valid, w_valid] = rec_valid

        self.last_extracted_records = output_tensor
        self.last_feat_16d_flat = feat_16d_flat

        # 3. Log JSON data once every 100 batches exclusively during the first epoch (current_epoch == 0)
        if self.current_epoch == 0:
            self.log_counter += 1
            if self.log_counter % 100 == 0:
                with torch.no_grad():
                    b0_records = output_tensor[0] # (20, 20)
                    valid_boxes = []
                    for row in b0_records:
                        if row[0].item() != -1.0:  # Exclude padding rows (-1.0)
                            x1, y1, x2, y2 = row[:4].cpu().tolist()
                            vec = row[4:].cpu().tolist()
                            valid_boxes.append({
                                "box": [x1, y1, x2, y2],
                                "concept_vector_16d": [round(v, 4) for v in vec]
                            })
                    
                    json_log_data = {
                        "epoch": self.current_epoch + 1,
                        "step_or_batch_count": self.log_counter,
                        "detected_objects_count": len(valid_boxes),
                        "objects": valid_boxes
                    }
                    
                    file_path = os.path.join(self.output_dir, f"extraction_epoch1_step_{self.log_counter:06d}.json")
                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(json_log_data, f, ensure_ascii=False, indent=2)
                    
                    tqdm.write(f"💾 [Epoch 1 JSON Log Saved] {file_path} (Detected: {len(valid_boxes)} objects)")

        return output_tensor


class SpatialConceptNN(nn.Module):
    def __init__(self, filter_path, dict_path, json_path, default_labels, output_dir="space/output", num_classes=10, stride=2):
        super(SpatialConceptNN, self).__init__()
        self.extraction_layer = GeometricExtractionLayer(
            filter_path, dict_path, json_path, default_labels, output_dir=output_dir, stride=1, max_boxes=20
        )
        
        self.features = nn.Sequential(
            self.extraction_layer,
            nn.Linear(20, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def set_epoch(self, epoch):
        """Propagates the current epoch index to the internal feature extraction layer."""
        self.extraction_layer.current_epoch = epoch

    def forward(self, x):
        x = self.features(x) # (B, 20, 128)
        x, _ = torch.max(x, dim=1) # (B, 128)
        return self.classifier(x)


class MNISTPipeline:
    def __init__(self, device=None, config=None, stride=2):
        self.device = torch.device(
            device if device and torch.cuda.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.config = config
        self.space_dir = config.space_dir if config and hasattr(config, "space_dir") else "space"
        self.output_dir = os.path.join(self.space_dir, "output")
        os.makedirs(self.space_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        self.ckpt_filter = os.path.join(self.space_dir, "checkpoint_geometric_filter.pth")
        self.ckpt_dict = os.path.join(self.space_dir, "checkpoint_concept_dictionary.pth")
        self.ckpt_classifier = os.path.join(self.space_dir, "checkpoint_mnist_spatial_nn.pth")
        self.labels_json_path = os.path.join(self.space_dir, "concept_labels.json")
        self.stride = stride

        self.concept_labels = [
            "vertical_line", "horizontal_line", "l_shape", "t_shape", "cross_shape", "corner",
            "half_circle", "curve_left", "curve_right", "curve_up", "curve_down",
            "s_curve", "c_curve",
            "angle_30", "angle_60", "angle_neg30", "angle_neg60", "angle_45", "angle_neg45",
            "angle_120", "angle_135", "angle_neg120", "angle_neg135",
            "closed_loop", "upper_loop", "lower_loop", "left_loop", "right_loop",
            "endpoint_top", "endpoint_bottom", "endpoint_left", "endpoint_right"
        ]
        if os.path.exists(self.labels_json_path):
            try:
                with open(self.labels_json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.concept_labels = data.get("labels", self.concept_labels)
            except Exception:
                pass

    def _get_mnist_data(self):
        loader = MNISTDataLoader(config=self.config)
        loader._ensure_train_loaded()
        loader._ensure_test_loaded()
        return (loader._X_train, loader._y_train), (loader._X_test, loader._y_test)

    def train(self):
        print("\n=== [MNIST Train] Pure Integrated Model Training ===")
        if not os.path.exists(self.ckpt_filter) or not os.path.exists(self.ckpt_dict):
            print("[ERROR] Checkpoints for geometric filters or dictionary model not found. Please run action [1] first.")
            return

        (X_train, y_train), _ = self._get_mnist_data()
        
        configured_max = getattr(self.config.app, "max_train_samples", -1) if (self.config and hasattr(self.config, "app")) else -1
        max_samples = len(X_train) if configured_max <= 0 else min(configured_max, len(X_train))
        
        print(f"💡 Training with {max_samples} samples (Configured max: {configured_max}).")
        
        X_sub = X_train[:max_samples].astype(np.float32)
        if X_sub.max() > 1.0:
            X_sub /= 255.0

        X_tensor = torch.tensor(X_sub, dtype=torch.float32).unsqueeze(1)
        y_tensor = torch.tensor(y_train[:max_samples], dtype=torch.long)

        dataloader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=64, shuffle=True)
        
        model = SpatialConceptNN(
            self.ckpt_filter, self.ckpt_dict, self.labels_json_path, self.concept_labels, output_dir=self.output_dir, num_classes=10, stride=self.stride
        ).to(self.device)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        model.train()
        epochs = 10
        for epoch in range(epochs):
            # Notify the model of the current epoch to enable logging only on the first epoch
            model.set_epoch(epoch)
            
            total_loss, correct, total = 0.0, 0, 0
            pbar = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{epochs}]", ncols=100)
            
            for batch_x, batch_y in pbar:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * batch_x.size(0)
                _, preds = outputs.max(1)
                correct += preds.eq(batch_y).sum().item()
                total += batch_y.size(0)
                pbar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{100.0 * correct / total:.2f}%")

            tqdm.write(f"👉 Epoch {epoch+1} Completed | Loss: {total_loss/total:.4f} | Accuracy: {100.0 * correct / total:.2f}%")

        torch.save({"model_state_dict": model.state_dict()}, self.ckpt_classifier)
        print(f"✅ Integrated classification model saved to: {self.ckpt_classifier}")

    def test(self):
        print("\n=== [MNIST Test] Evaluation ===")
        if not os.path.exists(self.ckpt_classifier):
            print("[ERROR] Classifier model checkpoint not found.")
            return

        _, (X_test, y_test) = self._get_mnist_data()

        configured_max = getattr(self.config.app, "max_test_samples", -1) if (self.config and hasattr(self.config, "app")) else -1
        test_samples = len(X_test) if configured_max <= 0 else min(configured_max, len(X_test))

        X_test_sub = X_test[:test_samples].astype(np.float32)
        if X_test_sub.max() > 1.0:
            X_test_sub /= 255.0

        X_tensor = torch.tensor(X_test_sub, dtype=torch.float32).unsqueeze(1)
        y_tensor = torch.tensor(y_test[:test_samples], dtype=torch.long)
        dataloader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=64, shuffle=False)

        model = SpatialConceptNN(
            self.ckpt_filter, self.ckpt_dict, self.labels_json_path, self.concept_labels, output_dir=self.output_dir, num_classes=10, stride=self.stride
        ).to(self.device)
        
        ckpt = torch.load(self.ckpt_classifier, map_location=self.device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_y in tqdm(dataloader, desc="Evaluating", ncols=100):
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = model(batch_x)
                _, preds = outputs.max(1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        correct = (all_preds == all_targets).sum()
        total = len(all_targets)
        overall_acc = 100.0 * correct / total

        print(f"\n📊 [Test Results] Overall Accuracy: {overall_acc:.2f}% (Evaluated samples: {total})")

        # Display per-class detailed accuracy for MNIST digits 0-9
        print("\n--- Per-Class Test Accuracy (MNIST Digits 0-9) ---")
        for digit in range(10):
            mask = (all_targets == digit)
            if mask.sum() > 0:
                digit_acc = (all_preds[mask] == digit).sum() / mask.sum() * 100.0
                print(f"  Digit {digit} : {digit_acc:6.2f}%  ({mask.sum()} samples)")

    def train_data_eval(self):
        TrainDataEvaluator(self).execute()