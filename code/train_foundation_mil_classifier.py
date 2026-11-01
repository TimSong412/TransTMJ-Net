#!/usr/bin/env python3
"""Model inference and evaluation helpers. Use run_evaluation.py as the CLI."""

from __future__ import annotations
import contextlib
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, confusion_matrix, f1_score, precision_score, roc_auc_score
from torch.utils.data import DataLoader
from transformers import Dinov2Model, SiglipVisionModel


class LoRALinear(nn.Module):
    """A minimal LoRA adapter that keeps the public base linear layer frozen."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.base.requires_grad_(False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base_output = self.base(inputs)
        delta = self.lora_B(self.lora_A(self.dropout(inputs)))
        return base_output + delta * self.scaling


def inject_lora(
    model: nn.Module,
    target_module_suffixes: tuple[str, ...],
    module_name_contains: str,
    rank: int,
    alpha: float,
    dropout: float,
) -> list[str]:
    replacements: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if (
            isinstance(module, nn.Linear)
            and module_name_contains in name
            and any(name.endswith(suffix) for suffix in target_module_suffixes)
        ):
            replacements.append((name, module))
    if not replacements:
        raise RuntimeError(f"No LoRA targets found for {target_module_suffixes}")
    replaced = []
    for name, module in replacements:
        parent_name, leaf_name = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        setattr(parent, leaf_name, LoRALinear(module, rank, alpha, dropout))
        replaced.append(name)
    return replaced


class FoundationMILClassifier(nn.Module):
    def __init__(
        self,
        backbone_path: Path,
        model_family: str,
        lora_rank: int,
        lora_alpha: float,
        lora_dropout: float,
        lora_targets: tuple[str, ...],
        attention_hidden: int,
        head_dropout: float,
        slice_chunk_size: int,
        gradient_checkpointing: bool,
        slice_feature_source: str,
        expected_slices: int,
        spatial_view_count: int,
        slice_view_position_embedding: bool,
        binary_head_mode: str,
        input_adapter_hidden: int,
        unfreeze_last_blocks: int,
        volume_transformer_layers: int,
        volume_transformer_heads: int,
        volume_transformer_ffn: int,
    ) -> None:
        super().__init__()
        self.backbone_path = str(backbone_path)
        if model_family == "siglip":
            model_class = SiglipVisionModel
            encoder_name_fragment = ".encoder.layers."
        elif model_family == "dinov2":
            model_class = Dinov2Model
            # Dinov2Model is the vision encoder itself, so its module names start
            # with ``encoder.layer`` rather than ``vision_model.encoder.layer``.
            encoder_name_fragment = "encoder.layer."
        else:
            raise ValueError(f"Unsupported model family: {model_family}")
        self.model_family = model_family
        self.input_adapter_hidden = input_adapter_hidden
        if input_adapter_hidden < 0:
            raise ValueError("input_adapter_hidden must be non-negative")
        self.input_adapter = (
            nn.Sequential(
                nn.Conv2d(3, input_adapter_hidden, kernel_size=3, padding=1),
                nn.GroupNorm(1, input_adapter_hidden),
                nn.GELU(),
                nn.Conv2d(input_adapter_hidden, 3, kernel_size=3, padding=1),
            )
            if input_adapter_hidden > 0
            else None
        )
        if self.input_adapter is not None:
            nn.init.zeros_(self.input_adapter[-1].weight)
            nn.init.zeros_(self.input_adapter[-1].bias)
        self.vision = model_class.from_pretrained(
            backbone_path, local_files_only=True, torch_dtype=torch.float32
        )
        self.vision.requires_grad_(False)
        self.lora_module_names = inject_lora(
            self.vision,
            lora_targets,
            encoder_name_fragment,
            lora_rank,
            lora_alpha,
            lora_dropout,
        )
        expected_lora_modules = int(self.vision.config.num_hidden_layers) * len(lora_targets)
        if len(self.lora_module_names) != expected_lora_modules:
            raise RuntimeError(
                f"Expected {expected_lora_modules} encoder LoRA modules, "
                f"found {len(self.lora_module_names)}"
            )
        if not 0 <= unfreeze_last_blocks <= int(self.vision.config.num_hidden_layers):
            raise ValueError("unfreeze_last_blocks is outside the encoder depth")
        self.unfreeze_last_blocks = unfreeze_last_blocks
        if unfreeze_last_blocks:
            if model_family == "siglip":
                encoder_layers = self.vision.vision_model.encoder.layers
            else:
                encoder_layers = self.vision.encoder.layer
            for layer in encoder_layers[-unfreeze_last_blocks:]:
                layer.requires_grad_(True)
        if gradient_checkpointing:
            self.vision.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            self.vision.enable_input_require_grads()
        feature_dim = int(self.vision.config.hidden_size)
        self.feature_dim = feature_dim
        self.slice_chunk_size = slice_chunk_size
        self.slice_feature_source = slice_feature_source
        self.expected_slices = expected_slices
        self.spatial_view_count = spatial_view_count
        self.slice_view_position_embedding = (
            nn.Parameter(torch.empty(1, spatial_view_count, expected_slices, feature_dim))
            if slice_view_position_embedding
            else None
        )
        if self.slice_view_position_embedding is not None:
            nn.init.trunc_normal_(self.slice_view_position_embedding, std=0.02)
        if volume_transformer_layers < 0:
            raise ValueError("volume_transformer_layers must be non-negative")
        if volume_transformer_heads <= 0 or feature_dim % volume_transformer_heads:
            raise ValueError("volume_transformer_heads must divide the feature dimension")
        if volume_transformer_ffn <= 0:
            raise ValueError("volume_transformer_ffn must be positive")
        self.volume_transformer_layers = volume_transformer_layers
        self.volume_transformer_heads = volume_transformer_heads
        self.volume_transformer_ffn = volume_transformer_ffn
        self.volume_transformer = (
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=feature_dim,
                    nhead=volume_transformer_heads,
                    dim_feedforward=volume_transformer_ffn,
                    dropout=head_dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                ),
                num_layers=volume_transformer_layers,
                enable_nested_tensor=False,
            )
            if volume_transformer_layers
            else None
        )
        self.slice_feature_projection = (
            nn.Sequential(
                nn.LayerNorm(feature_dim * 2),
                nn.Linear(feature_dim * 2, feature_dim),
                nn.GELU(),
            )
            if slice_feature_source == "pooler_mean_concat"
            else None
        )
        # Keep this module present for every feature-source mode. This permits
        # controlled pooler/patch feature ablations without changing the model
        # architecture, total parameter count, or checkpoint key set.
        self.patch_attention = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, attention_hidden),
            nn.Tanh(),
            nn.Linear(attention_hidden, 1),
        )
        self.slice_attention = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, attention_hidden),
            nn.Tanh(),
            nn.Linear(attention_hidden, 1),
        )
        self.pool_projection = nn.Sequential(
            nn.LayerNorm(feature_dim * 2),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.GELU(),
            nn.Dropout(head_dropout),
        )
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.binary_head_mode = binary_head_mode
        # Keep the independent head instantiated in both modes so architecture
        # and parameter count remain identical across endpoint formulations.
        self.binary_head = nn.Linear(feature_dim, 1)
        self.three_class_head = nn.Linear(feature_dim, 3)

    def forward_features(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, slices, channels, height, width = images.shape
        flat = images.reshape(batch * slices, channels, height, width)
        if self.input_adapter is not None:
            flat = flat + self.input_adapter(flat)
        chunks = []
        for start in range(0, len(flat), self.slice_chunk_size):
            output = self.vision(pixel_values=flat[start : start + self.slice_chunk_size])
            pooled = getattr(output, "pooler_output", None)
            dense_tokens = output.last_hidden_state
            if self.model_family == "dinov2":
                dense_tokens = dense_tokens[:, 1:]
            dense_mean = dense_tokens.mean(dim=1)
            if pooled is None:
                pooled = output.last_hidden_state[:, 0]
            if self.slice_feature_source == "pooler":
                selected = pooled
            elif self.slice_feature_source == "mean_patches":
                selected = dense_mean
            elif self.slice_feature_source == "pooler_mean_average":
                selected = (pooled + dense_mean) * 0.5
            elif self.slice_feature_source == "pooler_mean_concat":
                assert self.slice_feature_projection is not None
                selected = self.slice_feature_projection(torch.cat((pooled, dense_mean), dim=-1))
            elif self.slice_feature_source == "learned_patch_attention":
                assert self.patch_attention is not None
                patch_weights = torch.softmax(
                    self.patch_attention(dense_tokens).squeeze(-1), dim=1
                )
                selected = (dense_tokens * patch_weights.unsqueeze(-1)).sum(dim=1)
            else:  # guarded by argparse; kept defensive for direct construction
                raise ValueError(f"Unknown slice feature source: {self.slice_feature_source}")
            chunks.append(selected)
        slice_features = torch.cat(chunks, dim=0).reshape(batch, slices, self.feature_dim)
        if self.slice_view_position_embedding is not None:
            expected = self.expected_slices * self.spatial_view_count
            position = self.slice_view_position_embedding.reshape(1, expected, self.feature_dim)
            if slices == expected * 2:
                # Feature-level CBCT+sMRI fusion reuses the same spatial/depth
                # positions for both modality token groups without adding params.
                position = torch.cat((position, position), dim=1)
            elif slices != expected:
                raise RuntimeError(
                    f"Expected {expected} or {expected * 2} slice/view tokens for "
                    f"position embedding, got {slices}"
                )
            slice_features = slice_features + position
        if self.volume_transformer is not None:
            slice_features = self.volume_transformer(slice_features)
        attention_weights = torch.softmax(self.slice_attention(slice_features).squeeze(-1), dim=1)
        attention_pool = (slice_features * attention_weights.unsqueeze(-1)).sum(dim=1)
        mean_pool = slice_features.mean(dim=1)
        features = self.pool_projection(torch.cat((mean_pool, attention_pool), dim=-1))
        return self.feature_norm(features), attention_weights

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features, attention_weights = self.forward_features(images)
        three_class_logits = self.three_class_head(features)
        binary_logits = (
            self.binary_head(features).squeeze(-1)
            if self.binary_head_mode == "independent"
            else three_class_logits[:, 1] - three_class_logits[:, 0]
        )
        return {
            "binary_logits": binary_logits,
            "three_class_logits": three_class_logits,
            "features": features,
            "slice_attention": attention_weights,
        }


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (probabilities >= edges[index]) & (probabilities <= edges[index + 1])
        else:
            mask = (probabilities >= edges[index]) & (probabilities < edges[index + 1])
        if mask.any():
            value += mask.mean() * abs(labels[mask].mean() - probabilities[mask].mean())
    return float(value)


def threshold_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(np.int64)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "sensitivity": float(tp / max(1, tp + fn)),
        "specificity": float(tn / max(1, tn + fp)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "threshold": float(threshold),
        "confusion_matrix": matrix.tolist(),
    }


def patient_cluster_bootstrap_auc_ci(
    labels: np.ndarray,
    probabilities: np.ndarray,
    patient_ids: np.ndarray,
    iterations: int = 2000,
    seed: int = 20260813,
) -> dict:
    """Return a percentile AUC CI while resampling paired joints by patient."""
    unique_patients = np.unique(patient_ids)
    patient_indices = {
        patient_id: np.flatnonzero(patient_ids == patient_id) for patient_id in unique_patients
    }
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(iterations):
        sampled_patients = rng.choice(unique_patients, size=len(unique_patients), replace=True)
        sampled_indices = np.concatenate(
            [patient_indices[patient_id] for patient_id in sampled_patients]
        )
        sampled_labels = labels[sampled_indices]
        if np.unique(sampled_labels).size != 2:
            continue
        values.append(roc_auc_score(sampled_labels, probabilities[sampled_indices]))
    if not values:
        raise RuntimeError("Patient-cluster bootstrap produced no two-class samples")
    lower, upper = np.percentile(np.asarray(values), [2.5, 97.5])
    return {
        "lower": float(lower),
        "upper": float(upper),
        "method": "patient-cluster percentile bootstrap",
        "iterations_requested": int(iterations),
        "iterations_valid": int(len(values)),
        "seed": int(seed),
        "patient_clusters": int(len(unique_patients)),
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    fixed_threshold: float | None = None,
) -> tuple[dict, dict, list[dict]]:
    model.eval()
    labels: list[np.ndarray] = []
    binary_probabilities: list[np.ndarray] = []
    three_probabilities: list[np.ndarray] = []
    joint_ids: list[str] = []
    patient_ids: list[str] = []
    attentions: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            # Keep native CUDA BF16 evaluation; MPS/CPU use FP32.
            inference_context = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if device.type == "cuda" and torch.cuda.is_bf16_supported()
                else contextlib.nullcontext()
            )
            with inference_context:
                output = model(images)
            binary_probabilities.append(torch.sigmoid(output["binary_logits"].float()).cpu().numpy())
            three_probabilities.append(torch.softmax(output["three_class_logits"].float(), dim=1).cpu().numpy())
            attentions.append(output["slice_attention"].float().cpu().numpy())
            labels.append(batch["label"].numpy())
            joint_ids.extend(batch["joint_id"])
            patient_ids.extend(batch["patient_id"])
    labels_3 = np.concatenate(labels)
    all_binary_probabilities = np.concatenate(binary_probabilities)
    probabilities_3 = np.concatenate(three_probabilities)
    attention = np.concatenate(attentions)
    binary_eligible = labels_3 != 2
    labels_binary = (labels_3[binary_eligible] == 1).astype(np.int64)
    probabilities = all_binary_probabilities[binary_eligible]
    binary_patient_ids = np.asarray(patient_ids)[binary_eligible]
    auc = float(roc_auc_score(labels_binary, probabilities))
    auc_ci = patient_cluster_bootstrap_auc_ci(
        labels_binary, probabilities, binary_patient_ids
    )
    if fixed_threshold is None:
        raise ValueError("A frozen threshold is required.")
    else:
        threshold = float(fixed_threshold)
        primary = threshold_metrics(labels_binary, probabilities, threshold)
        threshold_source = "locked_source_checkpoint"
    primary.update(
        {
            "n": int(len(labels_binary)),
            "positive_n": int(labels_binary.sum()),
            "negative_n": int((1 - labels_binary).sum()),
            "excluded_posterior_n": int((labels_3 == 2).sum()),
            "auc": auc,
            "auc_95_ci": auc_ci,
            "brier": float(brier_score_loss(labels_binary, probabilities)),
            "ece_10bin": expected_calibration_error(labels_binary, probabilities),
            "threshold_source": threshold_source,
        }
    )
    prediction_3 = probabilities_3.argmax(axis=1)
    three_class = {
        "n": int(len(labels_3)),
        "accuracy": float(accuracy_score(labels_3, prediction_3)),
        "balanced_accuracy": float(balanced_accuracy_score(labels_3, prediction_3)),
        "macro_f1": float(f1_score(labels_3, prediction_3, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(labels_3, prediction_3, labels=[0, 1, 2]).tolist(),
    }
    try:
        three_class["macro_ovr_auc"] = float(
            roc_auc_score(labels_3, probabilities_3, multi_class="ovr", average="macro")
        )
    except ValueError:
        three_class["macro_ovr_auc"] = None
    records = []
    for joint_id, patient_id, label, probability, probability_3, weights in zip(
        joint_ids, patient_ids, labels_3, all_binary_probabilities, probabilities_3, attention
    ):
        eligible = int(label) != 2
        records.append(
            {
                "joint_id": joint_id,
                "patient_id": patient_id,
                "label": int(label),
                "primary_binary_eligible": eligible,
                "binary_label": int(label == 1) if eligible else None,
                "binary_probability": float(probability) if eligible else None,
                "binary_prediction": int(probability >= threshold) if eligible else None,
                "three_class_probabilities": [float(value) for value in probability_3],
                "slice_attention": [float(value) for value in weights],
            }
        )
    return primary, three_class, records


def load_trainable_checkpoint(
    model: nn.Module,
    checkpoint: Path,
    allow_partial_vision_unfreeze: bool = False,
) -> dict:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload["trainable_state"]
    current_trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    missing = sorted(current_trainable - set(state))
    extra = sorted(set(state) - current_trainable)
    allowed_missing_prefixes = (
        "vision.encoder.layer.",
        "vision.vision_model.encoder.layers.",
    )
    invalid_missing = [
        name
        for name in missing
        if not (
            allow_partial_vision_unfreeze
            and name.startswith(allowed_missing_prefixes)
        )
    ]
    if extra or invalid_missing:
        raise RuntimeError(f"Trainable checkpoint mismatch; missing={missing[:5]}, extra={extra[:5]}")
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected checkpoint keys: {incompatible.unexpected_keys}")
    return payload
