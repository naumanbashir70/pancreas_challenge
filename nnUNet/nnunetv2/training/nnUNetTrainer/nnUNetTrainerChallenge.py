

import os, json, glob, re, ast
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("TensorBoard not available")


def _safe(obj, name, default=None):
    return getattr(obj, name, default)


def _normalize_case_id(s) -> str:
    """Clean up case ID to match the label dictionary keys"""
    if not isinstance(s, str):
        s = str(s)
    s = s.replace(".nii.gz", "").replace(".nii", "")
    s = re.sub(r"_\d{4}$", "", s)
    return s.strip()


def extract_case_ids(data_dict) -> list:
    """FIXED: Proper case ID extraction with numpy array handling"""
    raw_keys = []
    
    for field in ("keys", "keys_unpadded", "case_ids", "case_id"):
        if field in data_dict and data_dict[field] is not None:
            val = data_dict[field]
            if isinstance(val, (list, tuple)):
                raw_keys.extend(val)
            else:
                raw_keys.append(val)
    
    props = data_dict.get("properties", None)
    if isinstance(props, list):
        for p in props:
            if isinstance(p, dict):
                for k in ("keys", "key", "case_id", "name"):
                    if k in p and p[k] is not None:
                        raw_keys.append(p[k])
    elif isinstance(props, dict):
        for k in ("keys", "key", "case_id", "name"):
            if k in props and props[k] is not None:
                raw_keys.append(props[k])
    
    all_case_ids = []
    
    for raw_key in raw_keys:
        if raw_key is None:
            continue
            
        if hasattr(raw_key, '__array__') or type(raw_key).__name__ == 'ndarray':
            try:
                arr = np.asarray(raw_key)
                if arr.ndim == 0:
                    all_case_ids.append(_normalize_case_id(str(arr.item())))
                else:
                    for item in arr.flat:
                        all_case_ids.append(_normalize_case_id(str(item)))
                continue
            except:
                pass
        
        key_str = str(raw_key).strip()
        
        if key_str.startswith("[") and key_str.endswith("]"):
            try:
                parsed = ast.literal_eval(key_str)
                if isinstance(parsed, (list, tuple, np.ndarray)):
                    for item in parsed:
                        all_case_ids.append(_normalize_case_id(str(item)))
                    continue
            except (ValueError, SyntaxError):
                pass
            
            inner = key_str[1:-1].strip()
            if inner:
                parts = re.split(r'\s+', inner)
                for part in parts:
                    if part:
                        clean_part = part.strip("'\"")
                        if clean_part:
                            all_case_ids.append(_normalize_case_id(clean_part))
                continue
        
        if 'quiz_' in key_str:
            pattern = r'quiz_\d+_\d+'
            matches = re.findall(pattern, key_str)
            if matches:
                for match in matches:
                    all_case_ids.append(_normalize_case_id(match))
                continue
        
        normalized = _normalize_case_id(key_str)
        if normalized:
            all_case_ids.append(normalized)
    
    seen = set()
    unique_ids = []
    for case_id in all_case_ids:
        if case_id and case_id not in seen:
            seen.add(case_id)
            unique_ids.append(case_id)
    
    return unique_ids


class CrossAttentionPooling(nn.Module):
    def __init__(self, embed_dim: int, query_num: int = 1, num_heads: int = 1, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.query_num = query_num
        self.q = nn.Parameter(torch.randn(query_num, embed_dim) * 0.02)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        N = D * H * W
        feat = x.view(B, C, N).transpose(1, 2)
        Q = self.q.unsqueeze(0).expand(B, -1, -1)
        Q = self.q_proj(Q)
        K = self.k_proj(feat)
        V = self.v_proj(feat)
        pooled, _ = self.mha(Q, K, V)
        pooled = pooled.mean(dim=1)
        return self.out_proj(pooled)


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class nnUNetTrainerChallenge(nnUNetTrainer):
    alpha_cls: float = 2.0
    label_smoothing: float = 0.1
    cls_dropout_p: float = 0.3

    class_weights = None
    _enc_feat = None
    _caseid_to_subtype = {}
    _val_ids = set()
    _cap = None
    cls_head = None
    _wb = None
    _tb_writer = None
    _last_seg_loss = None
    _last_cls_loss = None
    _last_total_loss = None
    focal_loss = None
    _final_validation_done = False

    def _load_caseid_to_subtype(self):
        self._caseid_to_subtype = {}
        try:
            ds_name = self.plans_manager.dataset_name
            prep_base = Path(os.environ["nnUNet_preprocessed"])
            sidecar = prep_base / ds_name / "subtype_labels.json"
            if sidecar.exists():
                raw_dict = json.load(open(sidecar))
                self._caseid_to_subtype = {_normalize_case_id(k): int(v) for k, v in raw_dict.items()}
                print(f"✓ Loaded {len(self._caseid_to_subtype)} case labels from subtype_labels.json")
                return
        except Exception as e:
            print(f"Could not load classification labels: {e}")
            self._caseid_to_subtype = {}

    def _compute_class_weights(self, n_classes: int = 3):
        """More aggressive class weighting"""
        try:
            splits_path = Path(self.preprocessed_dataset_folder) / "splits_final.json"
            splits = json.load(open(splits_path))
            fold_idx = int(getattr(self, "fold", 0))
            if isinstance(splits, list) and len(splits) > fold_idx:
                train_ids = set(_normalize_case_id(x) for x in splits[fold_idx].get("train", []))
            else:
                train_ids = set()
        except Exception:
            train_ids = set(self._caseid_to_subtype.keys())

        cnt = Counter([self._caseid_to_subtype[cid] for cid in train_ids if cid in self._caseid_to_subtype])
        print(f"✓ Training set class counts: {dict(cnt)}")
        
        total = sum(cnt.values())
        freq = np.array([cnt.get(i, 1.0) for i in range(n_classes)], dtype=float)
        
        weights = total / (n_classes * freq)
        weights[0] *= 1.2
        weights[2] *= 2.0
        
        self.class_weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        print(f"✓ Aggressive class weights: {weights}")

    def _load_val_ids(self):
        self._val_ids = set()
        try:
            splits_path = Path(self.preprocessed_dataset_folder) / "splits_final.json"
            if splits_path.exists():
                splits = json.load(open(splits_path))
                fold_idx = int(getattr(self, "fold", 0))
                if isinstance(splits, list) and len(splits) > fold_idx:
                    self._val_ids = set(_normalize_case_id(x) for x in splits[fold_idx].get("val", []))
                print(f"✓ Loaded {len(self._val_ids)} validation IDs")
        except Exception as e:
            print(f"Could not load val IDs: {e}")

    def _find_deepest_encoder_block_and_channels(self):
        net = self.network
        plans_cfg = self.plans_manager.plans["configurations"][self.configuration_name]
        arch_kwargs = plans_cfg["architecture"]["arch_kwargs"]
        deepest_c = int(arch_kwargs["features_per_stage"][-1])

        target_name = None
        for name, mod in net.named_modules():
            if "encoder" in name and isinstance(mod, nn.Conv3d) and mod.out_channels == deepest_c:
                target_name = name.rsplit(".", 1)[0]
        if target_name is None:
            for name, mod in net.named_modules():
                if "encoder" in name and isinstance(mod, nn.Conv3d):
                    target_name = name.rsplit(".", 1)[0]
        if target_name is None:
            raise RuntimeError("Could not locate an encoder block to hook.")
        print(f"✓ Encoder tap: {target_name} with {deepest_c} ch")
        return dict(net.named_modules())[target_name], deepest_c

    def _init_classification_head(self, last_ch: int):
        """More powerful classification head with LayerNorm"""
        hidden_dim = last_ch
        
        self.cls_head = nn.Sequential(
            nn.Dropout(self.cls_dropout_p),
            nn.Linear(last_ch, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.cls_dropout_p),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(self.cls_dropout_p / 2),
            nn.Linear(hidden_dim // 2, 3)
        )
        
        for m in self.cls_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        nn.init.normal_(self.cls_head[-1].weight, mean=0.0, std=0.01)
        nn.init.constant_(self.cls_head[-1].bias, 0.0)
        
        if torch.cuda.is_available():
            self.cls_head.to(self.device)
        
        print(f"✓ Enhanced classification head: {last_ch} → {hidden_dim} → {hidden_dim//2} → 3")

    def _attach_encoder_hook_and_heads(self):
        target_module, last_ch = self._find_deepest_encoder_block_and_channels()

        def _hook(module, inp, out):
            self._enc_feat = out if isinstance(out, torch.Tensor) else out[0]
        target_module.register_forward_hook(_hook)
        print("✓ Encoder hook registered")

        cap_q = int(os.environ.get("MT_CAP_Q", "1"))
        cap_drop = float(os.environ.get("MT_CAP_DROP", "0.0"))
        wanted_heads = int(os.environ.get("MT_CAP_HEADS", "4"))
        heads = max(1, min(wanted_heads, last_ch))
        while last_ch % heads != 0 and heads > 1:
            heads //= 2
        heads = max(1, heads)

        self._cap = CrossAttentionPooling(embed_dim=last_ch, query_num=cap_q, num_heads=heads, dropout=cap_drop)
        
        self._init_classification_head(last_ch)
        
        if torch.cuda.is_available():
            self._cap.to(self.device)
        print(f"✓ CAP ready: C={last_ch}, Q={cap_q}, heads={heads}")

    def on_train_start(self):
        super().on_train_start()
        self.alpha_cls = getattr(self, "alpha_cls", 2.0)
        self.label_smoothing = getattr(self, "label_smoothing", 0.1)
        self.cls_dropout_p = getattr(self, "cls_dropout_p", 0.3)
        self._enc_feat = None
        self._caseid_to_subtype = {}
        self._cap = None
        self.cls_head = None
        self._last_seg_loss = None
        self._last_cls_loss = None
        self._last_total_loss = None
        self._final_validation_done = False

        self._load_caseid_to_subtype()
        self._compute_class_weights()
        self._attach_encoder_hook_and_heads()
        self._load_val_ids()

        self.focal_loss = FocalLoss(alpha=self.class_weights, gamma=2.0)
        
        ds = _safe(self.plans_manager, "dataset_name", "unknown")
        cfg_name = _safe(self, "configuration_name", "unknown")
        fold = int(_safe(self, "fold", -1))

        # Initialize WandB
        try:
            self._wb = wandb.init(
                project=os.environ.get("WANDB_PROJECT", "nnUNet_Challenge"),
                name=f"ds{ds}_{cfg_name}_fold{fold}_MT_TrainingCurves",
                config={
                    "dataset": ds,
                    "config": cfg_name,
                    "fold": fold,
                    "alpha_cls": self.alpha_cls,
                    "focal_loss": True,
                    "metrics_reloaded": True,
                }
            )
            wandb.define_metric("iter")
            wandb.define_metric("epoch")
            wandb.define_metric("train/*", step_metric="iter")
            wandb.define_metric("val/*", step_metric="epoch")
            print("✓ W&B initialized with training curves")
        except Exception as e:
            print(f"W&B init: {e}")

        # Initialize TensorBoard
        if TENSORBOARD_AVAILABLE:
            try:
                tb_dir = Path(self.output_folder) / "tensorboard"
                tb_dir.mkdir(exist_ok=True)
                self._tb_writer = SummaryWriter(str(tb_dir))
                print(f"TensorBoard initialized at {tb_dir}")
            except Exception as e:
                print(f"TensorBoard init failed: {e}")
                self._tb_writer = None
        else:
            print("TensorBoard not available")

    def _build_cls_targets(self, case_ids):
        if not isinstance(case_ids, (list, tuple)):
            case_ids = [case_ids]
        
        targets = []
        for cid in case_ids:
            label = self._caseid_to_subtype.get(cid, 0)
            targets.append(label)
        
        return torch.tensor(targets, dtype=torch.long, device=self.device, requires_grad=False)

    def _compute_loss(self, output, target, data_dict):
        seg_logits = output
        if self._enc_feat is None:
            raise RuntimeError("Encoder hook failed.")
        enc = self._enc_feat
        self._enc_feat = None

  
        case_ids = extract_case_ids(data_dict)
        
      
        unique_cases = []
        case_to_indices = {}
        for idx, cid in enumerate(case_ids):
            if cid not in case_to_indices:
                unique_cases.append(cid)
                case_to_indices[cid] = []
            case_to_indices[cid].append(idx)

        feat_vec = self._cap(enc)
        
       
        case_features = []
        case_labels = []
        for cid in unique_cases:
            indices = case_to_indices[cid]
            case_feat = feat_vec[indices].mean(dim=0, keepdim=True)
            case_features.append(case_feat)
            label = self._caseid_to_subtype.get(cid, 0)
            case_labels.append(label)
        
        case_features = torch.cat(case_features, dim=0)
        case_labels = torch.tensor(case_labels, dtype=torch.long, device=self.device)
        
      
        cls_logits = self.cls_head(case_features)
        
    
        seg_loss = self.loss(seg_logits, target)
        
        
        cls_loss = self._cls_loss(cls_logits, case_labels)
        
        total_loss = seg_loss + self.alpha_cls * cls_loss

      
        self._last_seg_loss = float(seg_loss.detach().cpu())
        self._last_cls_loss = float(cls_loss.detach().cpu())
        self._last_total_loss = float(total_loss.detach().cpu())

        return total_loss

    def on_iteration_end(self):
        """Log training losses after each iteration"""
        super().on_iteration_end()
        
        try:
            it = int(getattr(self, "num_iterations", 0))
            
           
            if (self._last_seg_loss is not None and 
                self._last_cls_loss is not None and 
                self._last_total_loss is not None and
                it % 10 == 0):
                
               
                if self._wb:
                    wandb.log({
                        "iter": it,
                        "train/seg_loss": self._last_seg_loss,
                        "train/cls_loss": self._last_cls_loss,
                        "train/total_loss": self._last_total_loss,
                    }, step=it)
                
               
                if self._tb_writer:
                    self._tb_writer.add_scalar('train/seg_loss', self._last_seg_loss, it)
                    self._tb_writer.add_scalar('train/cls_loss', self._last_cls_loss, it)
                    self._tb_writer.add_scalar('train/total_loss', self._last_total_loss, it)
                    
               
                if it % 100 == 0:
                    print(f"  Iter {it}: seg_loss={self._last_seg_loss:.4f}, cls_loss={self._last_cls_loss:.4f}, total={self._last_total_loss:.4f}")
                    
        except Exception as e:
            
            if it % 100 == 0:
                print(f"Training logging error: {e}")

    def _parse_segmentation_metrics_reloaded(self, summary_path, include_hd_msd=True):
        """Parse comprehensive segmentation metrics from nnU-Net validation summary"""
        metrics = {}
        try:
            with open(summary_path, 'r') as f:
                summ = json.load(f)
            
            
            if "mean_dice" in summ:
                metrics["val/seg_dice"] = float(summ["mean_dice"])
            
            
            for key in summ.keys():
                if key.startswith("dice_") or key.startswith("Dice_"):
                    class_name = key.split("_", 1)[1]
                    metrics[f"val/seg_dice_{class_name}"] = float(summ[key])
            
           
            if include_hd_msd:
                if "mean_hausdorff_distance_95" in summ:
                    metrics["val/seg_hd95"] = float(summ["mean_hausdorff_distance_95"])
                elif "hd95" in summ:
                    metrics["val/seg_hd95"] = float(summ["hd95"])
                
                if "mean_surface_distance" in summ:
                    metrics["val/seg_msd"] = float(summ["mean_surface_distance"])
            
           
            for metric in ["precision", "recall", "specificity", "accuracy"]:
                if metric in summ:
                    metrics[f"val/seg_{metric}"] = float(summ[metric])
            
        except Exception as e:
            print(f"Could not parse segmentation metrics: {e}")
        
        return metrics

    def _val_segmentation_metrics_reloaded(self, include_hd_msd=True):
        """Enhanced segmentation metrics following Metrics Reloaded"""
        out = Path(self.output_folder)
        summaries = sorted(glob.glob(str(out / "validation*" / "summary.json")))
        
        if not summaries:
            return {}
        
        try:
            latest_summary = summaries[-1]
            metrics = self._parse_segmentation_metrics_reloaded(latest_summary, include_hd_msd)
            return metrics
        except Exception as e:
            print(f"Error loading segmentation metrics: {e}")
            return {}

    def _val_classification_metrics(self):
        """Enhanced classification metrics with class-wise reporting"""
        if not hasattr(self, "dataloader_val") or self.dataloader_val is None:
            return {}

        self.network.eval()
        self._cap.eval()
        self.cls_head.eval()

        case_features = {}
        case_labels = {}
        max_batches = 200
        batches = 0

        try:
            with torch.no_grad():
                for data_dict in self.dataloader_val:
                    batches += 1
                    batch_keys = extract_case_ids(data_dict)
                    
                    data = data_dict["data"].to(self.device, non_blocking=True)

                    self._enc_feat = None
                    _ = self.network(data)
                    if self._enc_feat is None:
                        continue

                    enc = self._enc_feat
                    self._enc_feat = None
                    feat_vec = self._cap(enc)
                    
                    for i, cid in enumerate(batch_keys):
                        if cid not in case_features:
                            case_features[cid] = []
                            case_labels[cid] = self._caseid_to_subtype.get(cid, 0)
                        case_features[cid].append(feat_vec[i].detach().cpu())

                    if batches >= max_batches:
                        break
        finally:
            self.network.train()
            self._cap.train()
            self.cls_head.train()

        y_true, y_pred, y_probs = [], [], []
        
        for cid, features_list in case_features.items():
            if len(features_list) == 0:
                continue
                
            avg_features = torch.stack(features_list).mean(dim=0).to(self.device).unsqueeze(0)
            logits = self.cls_head(avg_features)
            probs = F.softmax(logits, dim=1)
            pred = torch.argmax(logits, dim=1).item()
            
            y_pred.append(pred)
            y_true.append(case_labels[cid])
            y_probs.append(probs.detach().cpu().numpy()[0])

        if len(y_true) == 0:
            return {}

        y_true = np.array(y_true, dtype=int)
        y_pred = np.array(y_pred, dtype=int)
        y_probs = np.array(y_probs)
        K = 3
        
       
        acc = float((y_true == y_pred).mean())
        eps = 1e-8
        
      
        cm = np.zeros((K, K), dtype=int)
        for t, p in zip(y_true, y_pred):
            if 0 <= t < K and 0 <= p < K:
                cm[t, p] += 1

 
        per_class_rec = []
        per_class_prec = []
        per_class_f1 = []
        
        for c in range(K):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            
            rec = tp / (tp + fn + eps)
            prec = tp / (tp + fp + eps)
            f1 = 2 * prec * rec / (prec + rec + eps)
            
            per_class_rec.append(rec)
            per_class_prec.append(prec)
            per_class_f1.append(f1)
        
        bal_acc = float(np.mean(per_class_rec))
        macro_f1 = float(np.mean(per_class_f1))
        
        
        from sklearn.metrics import roc_auc_score
        try:
            auc_roc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
        except:
            auc_roc = 0.0

        metrics = {
            "val/cls_accuracy": acc,
            "val/cls_balanced_accuracy": bal_acc,
            "val/cls_macro_f1": macro_f1,
            "val/cls_auc_roc": auc_roc,
            
         
            "val/cls_recall_0": per_class_rec[0],
            "val/cls_recall_1": per_class_rec[1],
            "val/cls_recall_2": per_class_rec[2],
            "val/cls_precision_0": per_class_prec[0],
            "val/cls_precision_1": per_class_prec[1],
            "val/cls_precision_2": per_class_prec[2],
            "val/cls_f1_0": per_class_f1[0],
            "val/cls_f1_1": per_class_f1[1],
            "val/cls_f1_2": per_class_f1[2],
        }
        

        try:
            import matplotlib.pyplot as plt
            from sklearn.metrics import ConfusionMatrixDisplay
            
            fig, ax = plt.subplots(figsize=(8, 6))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2])
            disp.plot(ax=ax, cmap='Blues')
            ax.set_title('Classification Confusion Matrix')
            metrics["val/cls_confusion_matrix"] = wandb.Image(fig)
            plt.close(fig)
        except Exception as e:
            print(f" Could not create confusion matrix: {e}")

        return metrics

    def on_epoch_end(self):
        epoch = int(_safe(self, "current_epoch", -1))
        logs = {"epoch": epoch}

        try:
            if getattr(self, "moving_average_losses", None) and len(self.moving_average_losses) > 0:
                epoch_loss = float(self.moving_average_losses[-1])
                logs["train/epoch_loss"] = epoch_loss
               
                if self._tb_writer:
                    self._tb_writer.add_scalar('train/epoch_loss', epoch_loss, epoch)
        except Exception:
            pass

        
        num_epochs = int(_safe(self, "num_epochs", 1000))
        is_final_epoch = (epoch == num_epochs - 1) or self._final_validation_done

      
        include_hd_msd = is_final_epoch
        seg_metrics = self._val_segmentation_metrics_reloaded(include_hd_msd=include_hd_msd)
        logs.update(seg_metrics)

        
        cls_metrics = self._val_classification_metrics()
        logs.update(cls_metrics)

      
        seg_dice = seg_metrics.get("val/seg_dice", 0)
        seg_hd95 = seg_metrics.get("val/seg_hd95", float('nan'))
        seg_msd = seg_metrics.get("val/seg_msd", float('nan'))
        
        cls_acc = cls_metrics.get("val/cls_accuracy", 0)
        cls_f1 = cls_metrics.get("val/cls_macro_f1", 0)
        cls_auc = cls_metrics.get("val/cls_auc_roc", 0)
        
        print(f"=== Epoch {epoch} Summary ===")
        if include_hd_msd:
            print(f"Segmentation: Dice={seg_dice:.3f}, HD95={seg_hd95:.3f}, MSD={seg_msd:.3f}")
        else:
            print(f"Segmentation: Dice={seg_dice:.3f}")
        print(f"Classification: Acc={cls_acc:.3f}, F1={cls_f1:.3f}, AUC={cls_auc:.3f}")

     
        if self._tb_writer:
            try:
                self._tb_writer.add_scalar('val/seg_dice', seg_dice, epoch)
                if include_hd_msd:
                    self._tb_writer.add_scalar('val/seg_hd95', seg_hd95, epoch)
                    self._tb_writer.add_scalar('val/seg_msd', seg_msd, epoch)
                self._tb_writer.add_scalar('val/cls_accuracy', cls_acc, epoch)
                self._tb_writer.add_scalar('val/cls_macro_f1', cls_f1, epoch)
                self._tb_writer.add_scalar('val/cls_auc_roc', cls_auc, epoch)
                self._tb_writer.flush()
            except Exception as e:
                print(f"TensorBoard logging error: {e}")

       
        if logs:
            try:
                wandb.log(logs, step=epoch)
                print(f"✓ Logged {len(logs)} metrics to WandB")
            except Exception as e:
                print(f" W&B logging error: {e}")

        return super().on_epoch_end()

    def on_train_end(self):
        """Perform final validation with HD and MSD metrics"""
        print("=== Performing final validation with HD and MSD calculation ===")
        self._final_validation_done = True
        
        
        final_logs = {}
        
      
        seg_metrics = self._val_segmentation_metrics_reloaded(include_hd_msd=True)
        final_logs.update(seg_metrics)

       
        cls_metrics = self._val_classification_metrics()
        final_logs.update(cls_metrics)

     
        seg_dice = seg_metrics.get("val/seg_dice", 0)
        seg_hd95 = seg_metrics.get("val/seg_hd95", float('nan'))
        seg_msd = seg_metrics.get("val/seg_msd", float('nan'))
        
        cls_acc = cls_metrics.get("val/cls_accuracy", 0)
        cls_f1 = cls_metrics.get("val/cls_macro_f1", 0)
        cls_auc = cls_metrics.get("val/cls_auc_roc", 0)
        
        print("=== FINAL VALIDATION SUMMARY ===")
        print(f"Segmentation: Dice={seg_dice:.3f}, HD95={seg_hd95:.3f}, MSD={seg_msd:.3f}")
        print(f"Classification: Acc={cls_acc:.3f}, F1={cls_f1:.3f}, AUC={cls_auc:.3f}")

       
        if final_logs and self._wb:
            try:
                wandb.log(final_logs, step=int(_safe(self, "current_epoch", -1)))
                print("✓ Logged final validation metrics to WandB")
            except Exception as e:
                print(f" Final W&B logging error: {e}")


        super().on_train_end()
        try:
            if self._wb:
                self._wb.finish()
            if self._tb_writer:
                self._tb_writer.close()
        except Exception:
            pass