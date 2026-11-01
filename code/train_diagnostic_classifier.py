#!/usr/bin/env python3
"""Model inference and evaluation helpers. Use run_evaluation.py as the CLI."""

from __future__ import annotations
import math
import random
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def robust01(volume: np.ndarray) -> np.ndarray:
    valid = volume[np.isfinite(volume)]
    nonzero = valid[np.abs(valid) > 1e-8]
    source = nonzero if nonzero.size >= 100 else valid
    low, high = np.percentile(source, [1, 99])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(volume, dtype=np.float32)
    return np.clip((volume - low) / (high - low), 0, 1).astype(np.float32)


def read_rgb01(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return (
            np.asarray(image.convert("RGB"), dtype=np.float32).transpose(2, 0, 1)
            / 255.0
        )


class JointDataset(Dataset):
    def __init__(
        self,
        items: list[dict],
        processed_root: Path,
        generated_root: Path | None,
        modality: str,
        slices: int,
        image_size: int,
        augment: bool,
        input_channels: int | None = None,
        roi_ratio: float = 1.0,
        aligned_reference: bool = False,
        depth_ratio: float = 1.0,
        flip_probability: float = 0.5,
        intensity_augment_strength: float = 0.1,
        normalization_mean: float | list[float] | tuple[float, ...] = 0.5,
        normalization_std: float | list[float] | tuple[float, ...] = 0.25,
        adjacent_slice_channels: bool = False,
        slice_jitter: int = 0,
        geometry_augment_probability: float = 0.0,
        rotate_degrees: float = 0.0,
        translate_ratio: float = 0.0,
        scale_ratio: float = 0.0,
        gamma_ratio: float = 0.0,
        noise_std: float = 0.0,
        spatial_view_ratios: list[float] | tuple[float, ...] | None = None,
        canonicalize_right_side: bool = False,
        intensity_representation: str = "native",
        alignment_augment_probability: float = 0.0,
        swap_smri_sides: bool = False,
        smri_channel_mode: str = "native",
        fusion_channel_mode: str = "absolute_difference",
        mirror_all: bool = False,
        adjacent_channel_order: str = "native",
        smri_archive_root: Path | None = None,
        modality_archive_root: Path | None = None,
        archive_reverse_depth: bool = False,
        smri_archive_subdirectory: str = "generate",
        smri_archive_normalization: str = "robust01",
    ) -> None:
        self.items = items
        self.processed_root = processed_root
        self.generated_root = generated_root
        self.smri_archive_root = smri_archive_root
        self.modality_archive_root = modality_archive_root
        self.archive_reverse_depth = archive_reverse_depth
        self.smri_archive_subdirectory = smri_archive_subdirectory
        self.smri_archive_normalization = smri_archive_normalization
        self.modality = modality
        self.slices = slices
        self.image_size = image_size
        self.augment = augment
        self.input_channels = input_channels
        self.roi_ratio = roi_ratio
        self.aligned_reference = aligned_reference
        self.depth_ratio = depth_ratio
        self.flip_probability = flip_probability
        self.intensity_augment_strength = intensity_augment_strength
        self.normalization_mean = normalization_mean
        self.normalization_std = normalization_std
        self.adjacent_slice_channels = adjacent_slice_channels
        self.slice_jitter = slice_jitter
        self.geometry_augment_probability = geometry_augment_probability
        self.rotate_degrees = rotate_degrees
        self.translate_ratio = translate_ratio
        self.scale_ratio = scale_ratio
        self.gamma_ratio = gamma_ratio
        self.noise_std = noise_std
        self.spatial_view_ratios = (
            tuple(float(value) for value in spatial_view_ratios)
            if spatial_view_ratios is not None
            else None
        )
        self.canonicalize_right_side = canonicalize_right_side
        self.intensity_representation = intensity_representation
        self.alignment_augment_probability = alignment_augment_probability
        self.swap_smri_sides = swap_smri_sides
        self.smri_channel_mode = smri_channel_mode
        self.fusion_channel_mode = fusion_channel_mode
        self.mirror_all = mirror_all
        self.adjacent_channel_order = adjacent_channel_order
        if not 0 < roi_ratio <= 1:
            raise ValueError(f"roi_ratio must be in (0, 1], got {roi_ratio}")
        if self.spatial_view_ratios is not None:
            if not self.spatial_view_ratios:
                raise ValueError("spatial_view_ratios cannot be empty")
            if any(not 0 < value <= 1 for value in self.spatial_view_ratios):
                raise ValueError(
                    "spatial_view_ratios values must all be in (0, 1]"
                )
            if roi_ratio != 1.0:
                raise ValueError(
                    "roi_ratio must remain 1.0 when spatial_view_ratios are supplied"
                )
        if not 0 < depth_ratio <= 1:
            raise ValueError(f"depth_ratio must be in (0, 1], got {depth_ratio}")
        if not 0 <= flip_probability <= 1:
            raise ValueError(f"flip_probability must be in [0, 1], got {flip_probability}")
        if intensity_augment_strength < 0:
            raise ValueError("intensity_augment_strength must be non-negative")
        if adjacent_slice_channels and (
            (modality == "fusion" and fusion_channel_mode != "modality_tokens_adjacent")
            or input_channels != 3
        ):
            raise ValueError(
                "Adjacent-slice channels require one image modality or "
                "fusion_channel_mode=modality_tokens_adjacent, with input_channels=3"
            )
        if slice_jitter < 0:
            raise ValueError("slice_jitter must be non-negative")
        if not 0 <= geometry_augment_probability <= 1:
            raise ValueError("geometry_augment_probability must be in [0, 1]")
        if min(rotate_degrees, translate_ratio, scale_ratio, gamma_ratio, noise_std) < 0:
            raise ValueError("augmentation strengths must be non-negative")
        std_values = np.asarray(normalization_std, dtype=np.float32)
        if np.any(std_values <= 0):
            raise ValueError("normalization_std values must be positive")
        if intensity_representation not in ("native", "slice_standardized", "structural_rgb"):
            raise ValueError(f"Unsupported intensity representation: {intensity_representation}")
        if smri_channel_mode not in ("native", "first", "mean", "luminance"):
            raise ValueError(f"Unsupported sMRI channel mode: {smri_channel_mode}")
        if smri_archive_subdirectory not in ("generate", "shift"):
            raise ValueError(
                "smri_archive_subdirectory must be 'generate' or 'shift', got "
                f"{smri_archive_subdirectory!r}"
            )
        if smri_archive_normalization not in ("robust01", "native01"):
            raise ValueError(
                "smri_archive_normalization must be 'robust01' or 'native01', got "
                f"{smri_archive_normalization!r}"
            )
        if fusion_channel_mode not in (
            "absolute_difference",
            "mean",
            "signed_difference",
            "smri_emphasis",
            "cbct_emphasis",
            "modality_tokens_adjacent",
        ):
            raise ValueError(f"Unsupported fusion channel mode: {fusion_channel_mode}")
        if adjacent_channel_order not in (
            "native",
            "reverse_all",
            "reverse_left",
            "reverse_right",
        ):
            raise ValueError(f"Unsupported adjacent channel order: {adjacent_channel_order}")
        if adjacent_channel_order != "native" and not adjacent_slice_channels:
            raise ValueError(
                "Non-native adjacent channel order requires adjacent_slice_channels"
            )
        if intensity_representation == "structural_rgb" and (
            modality == "fusion" or input_channels != 3 or adjacent_slice_channels
        ):
            raise ValueError(
                "structural_rgb requires one grayscale modality, input_channels=3, "
                "and adjacent_slice_channels=False"
            )
        if not 0 <= alignment_augment_probability <= 1:
            raise ValueError("alignment_augment_probability must be in [0, 1]")
        if self.smri_archive_root is not None:
            if modality != "smri":
                raise ValueError("Archive sMRI input is supported only for the sMRI-only arm")
            if aligned_reference or alignment_augment_probability or swap_smri_sides:
                raise ValueError(
                    "Archive sMRI input forbids generated-pool alignment and side swapping"
                )
        if self.modality_archive_root is not None:
            if modality not in ("real_mri", "cbct", "fusion"):
                raise ValueError(
                    "Archive modality input is supported only for Real MRI, CBCT, or fusion"
                )
            if augment and modality != "fusion":
                raise ValueError("Archive modality input is evaluation-only")
            if aligned_reference or alignment_augment_probability:
                raise ValueError("Archive modality input forbids generated-pool alignment")
            if self.smri_archive_root is not None:
                raise ValueError("Cannot combine sMRI and modality archive roots")
        elif self.archive_reverse_depth and self.smri_archive_root is None:
            raise ValueError(
                "archive_reverse_depth requires an author-archive input root"
            )

    def __len__(self) -> int:
        return len(self.items)

    def _load(self, item: dict, kind: str) -> np.ndarray:
        source_joint_id = item.get("source_joint_id", item["joint_id"])
        case_dir = self.processed_root / source_joint_id
        use_aligned = self.aligned_reference or (
            self.augment
            and self.alignment_augment_probability > 0
            and random.random() < self.alignment_augment_probability
        )
        if self.modality_archive_root is not None:
            subdirectory = {
                "real_mri": "gt",
                "cbct": "condition",
                "smri": self.smri_archive_subdirectory,
            }.get(kind)
            if subdirectory is None:
                raise RuntimeError(f"Archive modality root cannot load {kind!r}")
            if self.modality != "fusion" and kind != self.modality:
                raise RuntimeError(
                    f"Archive {self.modality!r} dataset cannot load {kind!r}"
                )
            image_paths = sorted(
                (self.modality_archive_root / source_joint_id / subdirectory).glob(
                    "*.png"
                )
            )
            if not image_paths:
                raise FileNotFoundError(
                    f"No archive {subdirectory} PNGs for {item['joint_id']!r} under "
                    f"{self.modality_archive_root}"
                )
            # Archive fusion intentionally uses scalar condition/generate
            # volumes before applying the declared fusion channel mode.
            volume = np.stack(
                [read_rgb01(path).mean(axis=0) for path in image_paths], axis=0
            )
            if self.archive_reverse_depth:
                # The author archive enumerates gt/condition in the opposite
                # depth direction from the processed-DICOM training arrays.
                # Reverse only the slice order; pixels and intensity remain
                # the exact archive PNG data.
                volume = volume[::-1].copy()
            if kind == "smri" and self.smri_archive_normalization == "native01":
                volume = np.clip(volume, 0, 1)
            else:
                volume = (
                    robust01(volume)
                    if kind in ("real_mri", "smri")
                    else np.clip(volume, 0, 1)
                )
        elif kind == "cbct":
            reference = (
                self.generated_root / source_joint_id / "reference" / "cbct.npy"
                if self.generated_root is not None
                else Path("/__disabled_generated_reference__")
            )
            path = reference if use_aligned and reference.exists() else case_dir / "cbct.npy"
            volume = np.load(path).astype(np.float32)
            if use_aligned:
                if self.generated_root is None:
                    raise RuntimeError("Aligned CBCT requires --generated-root")
                generated_depth = len(np.load(self.generated_root / source_joint_id / "smri.npy", mmap_mode="r"))
                if len(volume) == generated_depth + 2:
                    volume = volume[1:-1]
            volume = np.clip(volume, 0, 1023) / 1023.0
        elif kind == "real_mri":
            reference = (
                self.generated_root / source_joint_id / "reference" / "real_mri.npy"
                if self.generated_root is not None
                else Path("/__disabled_generated_reference__")
            )
            path = reference if use_aligned and reference.exists() else case_dir / "real_mri.npy"
            volume = np.load(path).astype(np.float32)
            if use_aligned:
                if self.generated_root is None:
                    raise RuntimeError("Aligned real MRI requires --generated-root")
                generated_depth = len(np.load(self.generated_root / source_joint_id / "smri.npy", mmap_mode="r"))
                if len(volume) == generated_depth + 2:
                    volume = volume[1:-1]
            volume = (
                np.clip(volume, 0, 1)
                if self.smri_archive_root is not None
                and self.smri_archive_normalization == "native01"
                else robust01(volume)
            )
        elif kind == "smri":
            joint_id = source_joint_id
            if self.smri_archive_root is not None:
                image_paths = sorted(
                    (
                        self.smri_archive_root
                        / joint_id
                        / self.smri_archive_subdirectory
                    ).glob("*.png")
                )
                if not image_paths:
                    raise FileNotFoundError(
                        f"No author-archive {self.smri_archive_subdirectory} PNGs "
                        f"for {joint_id!r} under "
                        f"{self.smri_archive_root}"
                    )
                volume = np.stack(
                    [read_rgb01(path) for path in image_paths],
                    axis=0,
                )
                if self.archive_reverse_depth:
                    # Reverse slice order without changing PNG pixels or labels.
                    volume = volume[::-1].copy()
            else:
                if self.generated_root is None:
                    raise RuntimeError(
                        "sMRI input requires either an author-archive root or a "
                        "frozen-diffusion generated root"
                    )
                if self.swap_smri_sides:
                    if not joint_id.endswith(("_L", "_R")):
                        raise ValueError(
                            f"Cannot apply sMRI side correction to joint_id={joint_id!r}"
                        )
                    opposite_joint_id = joint_id[:-1] + (
                        "R" if joint_id.endswith("_L") else "L"
                    )
                    opposite_path = self.generated_root / opposite_joint_id / "smri.npy"
                    if opposite_path.exists():
                        joint_id = opposite_joint_id
                volume = np.load(self.generated_root / joint_id / "smri.npy").astype(np.float32)
            if self.swap_smri_sides and self.smri_archive_root is not None:
                raise RuntimeError("Archive sMRI side swapping is forbidden")
            if volume.ndim == 4:
                if volume.shape[1] != 3:
                    raise ValueError(
                        f"Expected decoder RGB volume [D,3,H,W], got {volume.shape}"
                    )
                if self.smri_channel_mode == "first":
                    volume = volume[:, 0]
                elif self.smri_channel_mode == "mean":
                    volume = volume.mean(axis=1)
                elif self.smri_channel_mode == "luminance":
                    weights = np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32)
                    volume = np.einsum("dchw,c->dhw", volume, weights)
            elif volume.ndim != 3:
                raise ValueError(f"Expected sMRI [D,H,W] or [D,3,H,W], got {volume.shape}")
            # ``native01`` applies to both supplied archive PNGs and the
            # regenerated-and-shifted Train NPY pool. Restricting it to archive
            # input would silently undo the paired-MRI brightness lock.
            volume = (
                np.clip(volume, 0, 1)
                if self.smri_archive_normalization == "native01"
                else robust01(volume)
            )
        else:
            raise ValueError(kind)
        if self.depth_ratio < 1:
            depth = len(volume)
            kept = min(depth, max(self.slices, round(depth * self.depth_ratio)))
            start = (depth - kept) // 2
            volume = volume[start : start + kept]
        centers = np.rint(np.linspace(0, len(volume) - 1, self.slices)).astype(int)
        if self.augment and self.slice_jitter:
            jitter = np.asarray(
                [random.randint(-self.slice_jitter, self.slice_jitter) for _ in centers]
            )
            centers = np.clip(centers + jitter, 0, len(volume) - 1)
        if self.adjacent_slice_channels:
            if volume.ndim != 3:
                raise ValueError(
                    "Adjacent-slice channels require a scalar sMRI representation; "
                    "select --smri-channel-mode first, mean, or luminance for decoder RGB."
                )
            indices = np.clip(centers[:, None] + np.asarray([-1, 0, 1])[None, :], 0, len(volume) - 1)
            return volume[indices]
        return volume[centers]

    def __getitem__(self, index: int) -> dict:
        item = self.items[index]
        if self.modality == "fusion":
            cbct = self._load(item, "cbct")
            smri = self._load(item, "smri")
            if cbct.shape[-2:] != smri.shape[-2:]:
                cbct_tensor = torch.from_numpy(cbct).float()
                if cbct_tensor.ndim == 3:
                    cbct_tensor = cbct_tensor[:, None]
                cbct = torch.nn.functional.interpolate(
                    cbct_tensor,
                    size=smri.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).numpy()
                if smri.ndim == 3:
                    cbct = cbct.squeeze(1)
            if self.fusion_channel_mode == "modality_tokens_adjacent":
                if cbct.ndim != 4 or smri.ndim != 4 or cbct.shape[1] != 3 or smri.shape[1] != 3:
                    raise ValueError(
                        "modality_tokens_adjacent requires adjacent RGB tensors for both modalities"
                    )
                # Encode CBCT and sMRI as separate slice tokens through one
                # shared vision tower. The model still emits one MIL prediction.
                array = np.concatenate((cbct, smri), axis=0)
            else:
                array = np.stack([cbct, smri], axis=1)
        else:
            loaded = self._load(item, self.modality)
            reverse_adjacent = self.adjacent_slice_channels and (
                self.adjacent_channel_order == "reverse_all"
                or (
                    self.adjacent_channel_order == "reverse_left"
                    and item.get("side") == "L"
                )
                or (
                    self.adjacent_channel_order == "reverse_right"
                    and item.get("side") == "R"
                )
            )
            if reverse_adjacent:
                loaded = loaded[:, ::-1].copy()
            array = loaded if loaded.ndim == 4 else loaded[:, None]
        tensor = torch.from_numpy(array).float()
        if self.mirror_all:
            tensor = torch.flip(tensor, dims=(-1,))
        if self.canonicalize_right_side and item.get("side") == "R":
            # Deterministically mirror right joints so left/right anatomy has
            # a common orientation. This is preprocessing, not random TTA.
            tensor = torch.flip(tensor, dims=(-1,))
        view_ratios = self.spatial_view_ratios or (self.roi_ratio,)
        spatial_views = []
        height, width = tensor.shape[-2:]
        for ratio in view_ratios:
            crop_h, crop_w = max(1, round(height * ratio)), max(1, round(width * ratio))
            top, left = (height - crop_h) // 2, (width - crop_w) // 2
            view = tensor[..., top : top + crop_h, left : left + crop_w]
            spatial_views.append(
                torch.nn.functional.interpolate(
                    view,
                    size=(self.image_size, self.image_size),
                    mode="bilinear",
                    align_corners=False,
                )
            )
        # Concatenating the jointly encoded views before MIL produces one
        # case-level prediction. This is a shared-weight multi-scale model,
        # not prediction-level ensembling or test-time augmentation.
        tensor = torch.cat(spatial_views, dim=0)
        if self.intensity_representation in ("slice_standardized", "structural_rgb"):
            mean = tensor.mean(dim=(-2, -1), keepdim=True)
            std = tensor.std(dim=(-2, -1), keepdim=True).clamp_min(1e-4)
            standardized = torch.sigmoid((tensor - mean) / std)
            if self.intensity_representation == "slice_standardized":
                tensor = standardized
            else:
                source = standardized
                padded = torch.nn.functional.pad(source, (1, 1, 1, 1), mode="replicate")
                center = padded[..., 1:-1, 1:-1]
                left = padded[..., 1:-1, :-2]
                right = padded[..., 1:-1, 2:]
                up = padded[..., :-2, 1:-1]
                down = padded[..., 2:, 1:-1]
                # Finite differences and a four-neighbour high-pass channel
                # retain morphology while avoiding a large CPU convolution in
                # every DataLoader worker.
                gradient_x = right - left
                gradient_y = down - up
                gradient = torch.sqrt(gradient_x.square() + gradient_y.square() + 1e-8)
                gradient_scale = gradient.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-4) * 4
                gradient = torch.clamp(gradient / gradient_scale, 0, 1)
                neighbour_mean = (center + left + right + up + down) / 5
                local_contrast = torch.abs(center - neighbour_mean)
                contrast_scale = (
                    local_contrast.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-4) * 4
                )
                local_contrast = torch.clamp(local_contrast / contrast_scale, 0, 1)
                tensor = torch.cat((source, gradient, local_contrast), dim=1)
        if self.input_channels is not None:
            if tensor.shape[1] == 1 and self.input_channels > 1:
                tensor = tensor.repeat(1, self.input_channels, 1, 1)
            elif (
                self.modality == "fusion"
                and tensor.shape[1] == 2
                and self.input_channels == 3
            ):
                cbct, smri = tensor[:, :1], tensor[:, 1:2]
                if self.fusion_channel_mode == "absolute_difference":
                    channels = (cbct, smri, torch.abs(cbct - smri))
                elif self.fusion_channel_mode == "mean":
                    channels = (cbct, smri, (cbct + smri) * 0.5)
                elif self.fusion_channel_mode == "signed_difference":
                    channels = (cbct, smri, torch.clamp((cbct - smri + 1) * 0.5, 0, 1))
                elif self.fusion_channel_mode == "smri_emphasis":
                    channels = (smri, smri, cbct)
                elif self.fusion_channel_mode == "cbct_emphasis":
                    channels = (cbct, cbct, smri)
                else:  # validated in __init__; defensive for mutated datasets
                    raise RuntimeError(
                        f"Unsupported fusion channel mode: {self.fusion_channel_mode}"
                    )
                # All modes feed one three-channel encoder and preserve exactly
                # the single-modality architecture/parameter count. They are
                # input encodings, not prediction-level ensembles.
                tensor = torch.cat(channels, dim=1)
            elif tensor.shape[1] != self.input_channels:
                raise ValueError(
                    f"Cannot map {tensor.shape[1]} modality channels to requested {self.input_channels}"
                )
        if self.augment:
            if random.random() < self.flip_probability:
                tensor = torch.flip(tensor, dims=(-1,))
            if random.random() < self.geometry_augment_probability:
                angle = math.radians(random.uniform(-self.rotate_degrees, self.rotate_degrees))
                scale = random.uniform(1 - self.scale_ratio, 1 + self.scale_ratio)
                tx = random.uniform(-self.translate_ratio, self.translate_ratio) * 2
                ty = random.uniform(-self.translate_ratio, self.translate_ratio) * 2
                theta = tensor.new_tensor(
                    [
                        [scale * math.cos(angle), -scale * math.sin(angle), tx],
                        [scale * math.sin(angle), scale * math.cos(angle), ty],
                    ]
                ).unsqueeze(0).repeat(len(tensor), 1, 1)
                grid = torch.nn.functional.affine_grid(theta, tensor.shape, align_corners=False)
                tensor = torch.nn.functional.grid_sample(
                    tensor, grid, mode="bilinear", padding_mode="zeros", align_corners=False
                )
            if self.intensity_augment_strength > 0:
                strength = self.intensity_augment_strength
                gain = random.uniform(1 - strength, 1 + strength)
                bias = random.uniform(-strength / 2, strength / 2)
                tensor = torch.clamp(tensor * gain + bias, 0, 1)
            if self.gamma_ratio > 0:
                gamma = random.uniform(1 - self.gamma_ratio, 1 + self.gamma_ratio)
                tensor = torch.clamp(tensor, 0, 1).pow(gamma)
            if self.noise_std > 0:
                tensor = torch.clamp(tensor + torch.randn_like(tensor) * self.noise_std, 0, 1)
        mean = torch.as_tensor(self.normalization_mean, dtype=tensor.dtype)
        std = torch.as_tensor(self.normalization_std, dtype=tensor.dtype)
        if mean.ndim == 0 or mean.numel() == 1:
            mean = mean.repeat(tensor.shape[1])
        if std.ndim == 0 or std.numel() == 1:
            std = std.repeat(tensor.shape[1])
        if mean.numel() != tensor.shape[1] or std.numel() != tensor.shape[1]:
            raise ValueError(
                f"Normalization has {mean.numel()}/{std.numel()} channels, "
                f"but input has {tensor.shape[1]}"
            )
        tensor = (tensor - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)
        return {
            "image": tensor,
            "label": torch.tensor(item["label"], dtype=torch.long),
            "joint_id": item["joint_id"],
            "patient_id": item["patient_id"],
        }
