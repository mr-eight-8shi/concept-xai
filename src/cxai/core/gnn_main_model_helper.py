import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from PIL import Image, ImageDraw


class TrainDataEvaluator:
    """
    Helper class to evaluate MNIST training data (first 1000 samples),
    extract Layer 1 geometric features (7x7), map them onto a 28x28 canvas,
    export coordinates, top-3 concept labels, and confidence scores as JSON,
    and save them as grid images in chunks of 10.
    """

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.device = pipeline.device
        self.space_dir = pipeline.space_dir
        self.output_dir = pipeline.output_dir  # Directly under the output directory
        self.stride = pipeline.stride
        self.ckpt_classifier = pipeline.ckpt_classifier
        self.ckpt_filter = pipeline.ckpt_filter
        self.ckpt_dict = pipeline.ckpt_dict
        self.labels_json_path = pipeline.labels_json_path
        self.concept_labels = pipeline.concept_labels
        self.label_images_dir = os.path.join(self.space_dir, "label_image")

    def execute(self):
        """
        Executes the overall evaluation pipeline (targeting up to the first 1000 samples).
        """
        print("\n=== [Train Data Eval] Layer 1 Geometric Feature Mapping & Top-3 JSON Export (Max 1000 samples) ===")
        if not os.path.exists(self.ckpt_classifier):
            print("[ERROR] Classifier model checkpoint not found.")
            return

        self._prepare_directory()
        X_sub, y_sub, eval_samples = self._load_evaluation_data()
        model = self._load_model()

        self._run_mapping_and_export(model, X_sub, y_sub, eval_samples)

    def _prepare_directory(self):
        """
        Initializes the output directory by clearing all existing files.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        for file in os.listdir(self.output_dir):
            file_path = os.path.join(self.output_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print(f"🧹 Cleared Output directory: {self.output_dir}")

    def _load_evaluation_data(self):
        """
        Retrieves MNIST training data for evaluation, limited to a maximum of 1000 samples.
        """
        (X_train, y_train), _ = self.pipeline._get_mnist_data()
        
        target_limit = 1000
        eval_samples = min(target_limit, len(X_train))

        X_sub = X_train[:eval_samples].astype(np.float32)
        y_sub = y_train[:eval_samples]
        if X_sub.max() > 1.0:
            X_sub /= 255.0

        return X_sub, y_sub, eval_samples

    def _load_model(self):
        """
        Loads the SpatialConceptNN model from saved checkpoints and sets it to evaluation mode.
        """
        from .gnn_main_model import SpatialConceptNN
        model = SpatialConceptNN(
            self.ckpt_filter, self.ckpt_dict, self.labels_json_path, self.concept_labels, num_classes=10, stride=self.stride
        ).to(self.device)
        
        ckpt = torch.load(self.ckpt_classifier, map_location=self.device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model

    def _flush_chunk_grid(self, records, batch_id):
        """
        Combines 10 records (raw image + embedded geometric feature image) into a single grid image and saves it.
        """
        if not records:
            return batch_id
        
        cols = 5  # 5 columns by 2 rows (10 items total)
        cell_w = 160
        cell_h = 190
        margin = 15
        
        n_items = len(records)
        n_rows = (n_items + cols - 1) // cols
        
        grid_w = cols * (cell_w + margin) + margin
        grid_h = n_rows * (cell_h + margin) + margin + 50
        
        grid_img = Image.new("RGB", (grid_w, grid_h), (25, 25, 25))
        draw = ImageDraw.Draw(grid_img)
        
        for idx_c, rec in enumerate(records):
            r = idx_c // cols
            c = idx_c % cols
            
            x_offset = margin + c * (cell_w + margin)
            y_offset = 45 + r * (cell_h + margin)
            
            draw.rectangle(
                [x_offset, y_offset, x_offset + cell_w, y_offset + cell_h],
                fill=(40, 40, 40), outline=(90, 90, 90), width=2
            )
            
            # Display raw image
            raw_disp = rec["raw_img"].resize((72, 72), Image.Resampling.NEAREST)
            grid_img.paste(raw_disp, (x_offset + 14, y_offset + 10))
            
            # Display geometric feature embedded image
            geo_disp = rec["geo_img"].resize((72, 72), Image.Resampling.NEAREST)
            grid_img.paste(geo_disp, (x_offset + 94, y_offset + 10))
            
            # Label text
            lbl_text = rec["filename_label"]
            draw.text((x_offset + 12, y_offset + 92), "Raw vs Geometric Layer", fill=(180, 180, 180))
            draw.text((x_offset + 12, y_offset + 112), lbl_text, fill=(240, 240, 240))
        
        grid_filename = os.path.join(self.output_dir, f"grid_batch_{batch_id:03d}.png")
        grid_img.save(grid_filename)
        print(f"  🖼️ Flushed 10-item grid image: {grid_filename}")
        return batch_id + 1

    def _run_mapping_and_export(self, model, X_sub, y_sub, eval_samples):
        """
        Scans all samples, extracts spatial gradients for the nn.Linear(20, 64) input layer,
        generates monochrome contrast images based on confidence scores and weighting formulas, and exports JSONs.
        """
        extractor = model.extraction_layer
        
        # Register a forward hook to capture the input tensor of the feature Linear layer
        linear_input_holder = {}
        target_linear_layer = None
        for name, module in model.features.named_modules():
            if isinstance(module, nn.Linear) and module.in_features == 20:
                target_linear_layer = module
                break

        def hook_fn(module, input, output):
            linear_input_holder["input"] = input[0]

        hook_handle = target_linear_layer.register_forward_hook(hook_fn) if target_linear_layer else None

        batch_records = []
        batch_id = 1

        print(f"🚀 Processing Linear-input gradient mapping across {eval_samples} samples...")
        
        try:
            for idx in tqdm(range(eval_samples), desc="Mapping & Exporting"):
                img_np = X_sub[idx]
                true_label = int(y_sub[idx])

                x_tensor = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
                
                # 1. Forward pass
                outputs = model(x_tensor)
                pred_label = int(outputs.argmax(dim=1).item())

                # 2. Extract input tensor of the Linear layer, enable gradients, and backpropagate
                canvas_np = np.zeros((28, 28), dtype=np.float32)
                detected_concepts_info = []
                
                parsed_boxes = []
                box_grads_list = []
                concept_grads_list = []

                if "input" in linear_input_holder and linear_input_holder["input"] is not None:
                    lin_in = linear_input_holder["input"].detach().clone().requires_grad_(True)
                    
                    x_feat = lin_in
                    feat_seq = list(model.features.children())
                    found_target = False
                    for layer in feat_seq:
                        if layer == target_linear_layer:
                            found_target = True
                        if found_target:
                            x_feat = layer(x_feat)
                    
                    x_pool, _ = torch.max(x_feat, dim=1)
                    final_logits = model.classifier(x_pool)
                    
                    target_score = final_logits[0, pred_label]
                    target_score.backward(retain_graph=True)
                    
                    lin_grads = lin_in.grad[0] # (20, 20)
                    records_tensor = extractor.last_extracted_records[0]  # (20, 20)

                    for i, row in enumerate(records_tensor):
                        x1, y1, x2, y2 = row[:4]
                        if x1.item() == -1.0:  # Skip padding rows
                            continue

                        concept_vec = row[4:].unsqueeze(0)  # (1, 16)
                        
                        box_grad_slice = lin_grads[i, :4]      # (4,)
                        concept_grad_slice = lin_grads[i, 4:]  # (16,)

                        max_box_grad = torch.max(torch.abs(box_grad_slice)).item()
                        max_concept_grad = torch.max(torch.abs(concept_grad_slice)).item()

                        # Compute Softmax probabilities via the concept dictionary model
                        logits = extractor.dict_model(concept_vec)
                        probs = torch.softmax(logits, dim=1)
                        
                        k_val = min(3, probs.size(1))
                        top_probs, top_indices = torch.topk(probs, k=k_val, dim=1)
                        
                        top_concepts_list = []
                        for p, cid in zip(top_probs[0], top_indices[0]):
                            cid_int = int(cid.item())
                            c_name = extractor.labels[cid_int]
                            confidence = float(p.item())
                            top_concepts_list.append({
                                "concept_label": c_name,
                                "confidence": round(confidence, 4)
                            })

                        top1_concept = top_concepts_list[0]
                        top1_cid_int = extractor.label_to_id[top1_concept["concept_label"]] if hasattr(extractor, "label_to_id") else extractor.labels.index(top1_concept["concept_label"])
                        top1_conf = top1_concept["confidence"]

                        box_grads_list.append(max_box_grad)
                        concept_grads_list.append(max_concept_grad)

                        parsed_boxes.append({
                            "x1": x1.item(), "y1": y1.item(), "x2": x2.item(), "y2": y2.item(),
                            "max_box_grad": max_box_grad,
                            "max_concept_grad": max_concept_grad,
                            "top1_cid_int": top1_cid_int,
                            "top1_conf": top1_conf,
                            "top_concepts_list": top_concepts_list
                        })

                # Calculate maximum contributions within the image to prevent division by zero
                max_pos_in_image = max(box_grads_list) if box_grads_list else 1.0
                max_concept_in_image = max(concept_grads_list) if concept_grads_list else 1.0
                if max_pos_in_image == 0: max_pos_in_image = 1.0
                if max_concept_in_image == 0: max_concept_in_image = 1.0

                # --- Map scores to the canvas using specific weighting formulas ---
                for box in parsed_boxes:
                    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
                    max_box_grad = box["max_box_grad"]
                    max_concept_grad = box["max_concept_grad"]
                    top1_cid_int = box["top1_cid_int"]
                    top1_conf = box["top1_conf"]
                    top_concepts_list = box["top_concepts_list"]

                    detected_concepts_info.append({
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2),
                        "max_box_grad": round(max_box_grad, 4),
                        "max_concept_grad": round(max_concept_grad, 4),
                        "top_concepts": top_concepts_list
                    })

                    # Search for the 7x7 reference image file
                    ref_img_np = None
                    if os.path.exists(self.label_images_dir):
                        for fname in os.listdir(self.label_images_dir):
                            if fname.startswith(f"{top1_cid_int:02d}_"):
                                ref_path = os.path.join(self.label_images_dir, fname)
                                if os.path.exists(ref_path):
                                    ref_pil = Image.open(ref_path).convert("L")
                                    ref_img_np = np.array(ref_pil, dtype=np.float32) / 255.0
                                    break

                    if ref_img_np is not None:
                        ix1, iy1 = int(max(0, x1)), int(max(0, y1))
                        ix2, iy2 = int(min(28, x2)), int(min(28, y2))
                        h_box, w_box = iy2 - iy1, ix2 - ix1

                        if h_box > 0 and w_box > 0:
                            resized_patch = np.array(
                                Image.fromarray((ref_img_np * 255).astype(np.uint8)).resize((w_box, h_box), Image.Resampling.NEAREST),
                                dtype=np.float32
                            ) / 255.0

                            pos_ratio = max_box_grad / max_pos_in_image
                            concept_ratio = max_concept_grad / max_concept_in_image

                            weighted_grad = (pos_ratio * 0.4) + (concept_ratio * 0.6)

                            if top1_conf >= 0.70:
                                conf_score = 1.0
                            else:
                                conf_score = top1_conf ** 2

                            raw_score = weighted_grad * conf_score

                            patch_gray = resized_patch * raw_score

                            sub_canvas = canvas_np[iy1:iy2, ix1:ix2]
                            canvas_np[iy1:iy2, ix1:ix2] = np.maximum(sub_canvas, patch_gray)

                # Prepare visualization images
                raw_img_pil = Image.fromarray((img_np * 255).astype(np.uint8)).convert("RGB")
                geo_canvas_small = Image.fromarray(np.clip(canvas_np * 255.0, 0, 255).astype(np.uint8))
                
                geo_canvas = geo_canvas_small.resize((112, 112), Image.Resampling.NEAREST).convert("RGB")
                raw_resized = raw_img_pil.resize((112, 112), Image.Resampling.NEAREST)

                batch_records.append({
                    "raw_img": raw_resized,
                    "geo_img": geo_canvas,
                    "filename_label": f"Id:{idx} T:{true_label} P:{pred_label}"
                })

                if len(batch_records) >= 10:
                    batch_id = self._flush_chunk_grid(batch_records, batch_id)
                    batch_records.clear()

                sample_json_data = {
                    "index": idx,
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "detected_concepts": detected_concepts_info
                }
                json_path = os.path.join(self.output_dir, f"sample_{idx}_true_{true_label}_pred_{pred_label}.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(sample_json_data, f, ensure_ascii=False, indent=4)

            if batch_records:
                self._flush_chunk_grid(batch_records, batch_id)

        finally:
            if hook_handle:
                hook_handle.remove()

        print(f"\n📊 [Evaluation & Export Completed]")
        print(f"  - Output files saved directly in: {self.output_dir}")