#!/usr/bin/env python3
"""Evaluate frozen classifiers with an external PNG directory; images are not bundled."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parent
ARMS = ("cbct", "smri", "fusion", "real_mri")
PATH_KEYS = ("manifest", "processed_root", "generated_root", "backbone_path",
             "checkpoint", "output_dir", "smri_archive_root",
             "smri_evaluation_archive_root", "evaluation_archive_root", "fusion_archive_root")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(value):
    if value is None:
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_config(path):
    cfg = read_json(path)
    if cfg.get("modality") not in ARMS or not cfg.get("eval_only"):
        raise ValueError("Use an eval-only configuration for one of the four packaged arms.")
    for key in PATH_KEYS:
        cfg[key] = resolve_path(cfg.get(key))
    return cfg


def archive_root(cfg):
    return (cfg["smri_evaluation_archive_root"] if cfg["modality"] == "smri"
            else cfg["fusion_archive_root"] if cfg["modality"] == "fusion"
            else cfg["evaluation_archive_root"])


def canonical_case_id(value):
    """Use English IDs while supporting existing external directory names."""
    for legacy, english in (
        ("\u914d\u5bf9\u540e3D\u6570\u636e", "paired_3D_data"),
        ("1-2\u671f", "phase1-2"),
        ("3\u671f", "phase3"),
        ("\u683c\u5c3c\u4e4c\u65af", "Genius"),
    ):
        value = value.replace(legacy, english)
    return value


def case_directories(root):
    result = {}
    for path in root.iterdir():
        if not path.is_dir():
            continue
        case_id = canonical_case_id(path.name)
        if case_id in result:
            raise ValueError("Multiple external directories resolve to the same case identifier.")
        result[case_id] = path
    return result


def check_inputs(cfg, verify_hashes=True):
    index = read_json(ROOT / "model_index.json")
    model = next(x for x in index["models"] if x["key"] == cfg["modality"])
    if not cfg["checkpoint"].is_file():
        raise FileNotFoundError(cfg["checkpoint"])
    if sha256(cfg["checkpoint"]) != model["checkpoint_sha256"]:
        raise ValueError("Checkpoint SHA256 differs from the locked model index.")
    for file in read_json(ROOT / "DINOv2_DOWNLOAD.json")["files"]:
        path = cfg["backbone_path"] / file["filename"]
        if not path.is_file() or path.stat().st_size != file["expected_bytes"]:
            raise ValueError(f"Missing or wrong-sized backbone file: {path}")
        if verify_hashes and sha256(path) != file["expected_sha256"]:
            raise ValueError(f"Backbone SHA256 mismatch: {path}")
    if cfg["manifest"] is None:
        raise ValueError("Private manifest required. 联系我们获取")
    bundle = read_json(cfg["manifest"])
    items, cohort_audit = validate_private_manifest(bundle)
    ids = {x["joint_id"] for x in items}
    root = archive_root(cfg)
    if root is None or not root.is_dir():
        raise FileNotFoundError(f"Missing evaluation archive: {root}")
    directories = case_directories(root)
    case_ids = set(directories)
    if len(case_ids) != 74 or not ids.issubset(case_ids):
        raise ValueError("Expected all 74 archive joints, including the 63 evaluated joints.")
    views = {"cbct": ["condition"], "smri": ["shift"],
             "fusion": ["condition", "shift"], "real_mri": ["gt"]}[cfg["modality"]]
    if cfg["modality"] in ("smri", "fusion") and cfg["smri_archive_subdirectory"] != "shift":
        raise ValueError("The packaged sMRI/fusion result requires the shift PNG view.")
    counts = {}
    for view in views:
        per_case = {case: len(list((directories[case] / view).glob("*.png"))) for case in case_ids}
        if sum(per_case.values()) != 1462 or any(not 15 <= n <= 25 for n in per_case.values()):
            raise ValueError(f"Invalid PNG slice inventory for {view}")
        counts[view] = sum(per_case[x] for x in ids)
    if cfg["modality"] == "fusion":
        for case in case_ids:
            a = {x.name for x in (directories[case] / "condition").glob("*.png")}
            b = {x.name for x in (directories[case] / "shift").glob("*.png")}
            if a != b:
                raise ValueError("Unpaired fusion slice names in the private input data.")
    return items, {"status": "PASS", "n_archive_joints": 74, "n_evaluation_joints": 63,
                   "n_patients": 60, "n_normal": 28, "n_anterior": 35,
                   "evaluation_png_counts": counts, "image_hashes_checked": False,
                   "data_source": "external_private_directory", **cohort_audit}


def validate_private_manifest(bundle):
    """Validate an authorized external cohort without bundling reference identities."""
    if not isinstance(bundle, dict) or not isinstance(bundle.get("items"), list):
        raise ValueError("An authorized external manifest with items is required. 联系我们获取")
    if bundle.get("synthetic_example_only"):
        raise ValueError("The fictional schema example cannot be used as patient data.")
    items = [dict(x) for x in bundle["items"] if x.get("split") == "test"]
    required = {"joint_id", "patient_id", "side", "label", "split"}
    if any(not required.issubset(x) for x in items):
        raise ValueError("Private test records are missing required fields.")
    if any(not isinstance(x["joint_id"], str) or not x["joint_id"] or
           not isinstance(x["patient_id"], str) or not x["patient_id"] or
           x["side"] not in ("L", "R") or type(x["label"]) is not int or x["label"] not in (0, 1)
           for x in items):
        raise ValueError("Invalid private identifier, side or binary label.")
    ids = {x["joint_id"] for x in items}
    if len(items) != 63 or len(ids) != 63 or len({x["patient_id"] for x in items}) != 60:
        raise ValueError("Expected the configured 63-joint, 60-ID cohort.")
    if sum(x["label"] == 0 for x in items) != 28 or sum(x["label"] == 1 for x in items) != 35:
        raise ValueError("Expected 28 normal and 35 anterior-displacement joints.")
    if any(Path(j).name != j or j in (".", "..") or "\\" in j for j in ids):
        raise ValueError("Joint identifiers must be single directory names.")
    # Canonical identities must come from a verified private source mapping.
    # Counts alone cannot authenticate membership or labels.
    return items, {
        "manifest_source": "user_supplied_private_manifest",
        "original_cohort_identity_verified": False,
        "identity_note": "Counts and format checked only; exact membership and labels require a private source audit.",
    }


def require_external_private_path(path):
    path = Path(path).expanduser().resolve()
    if path == ROOT or ROOT in path.parents:
        raise ValueError("Private inputs and outputs must be outside the distributable package.")
    return path


def write_private_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        os.chmod(path, 0o600)
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def write_private_predictions(directory, predictions, resolved_config):
    """Only called after --private-output-dir explicitly enables private export."""
    write_private_json(directory / "predictions.json", predictions)
    write_private_json(directory / "resolved_config.json", resolved_config)
    path = directory / "predictions.csv"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as stream:
        os.chmod(path, 0o600)
        keys = ["joint_id", "patient_id", "binary_label", "binary_probability", "binary_prediction"]
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: row[key] for key in keys} for row in predictions)


def make_dataset(cfg, items):
    from train_diagnostic_classifier import JointDataset
    directories = case_directories(archive_root(cfg))
    items = [dict(item, source_joint_id=directories[item["joint_id"]].name) for item in items]
    kwargs = {key: cfg[key] for key in inspect.signature(JointDataset).parameters if key in cfg}
    kwargs.update(items=items, augment=False, input_channels=3, generated_root=None,
                  flip_probability=0.0, intensity_augment_strength=0.0, slice_jitter=0,
                  geometry_augment_probability=0.0, rotate_degrees=0.0, translate_ratio=0.0,
                  scale_ratio=0.0, gamma_ratio=0.0, noise_std=0.0,
                  alignment_augment_probability=0.0,
                  smri_archive_root=archive_root(cfg) if cfg["modality"] == "smri" else None,
                  modality_archive_root=archive_root(cfg) if cfg["modality"] != "smri" else None,
                  archive_reverse_depth=cfg["evaluation_archive_reverse_depth"])
    return JointDataset(**kwargs)


def make_model(cfg):
    from train_foundation_mil_classifier import FoundationMILClassifier, load_trainable_checkpoint
    kwargs = {key: cfg[key] for key in inspect.signature(FoundationMILClassifier).parameters if key in cfg}
    kwargs.update(expected_slices=cfg["slices"],
                  spatial_view_count=len(cfg["spatial_view_ratios"] or [cfg["roi_ratio"]]),
                  lora_targets=tuple(cfg["lora_targets"]))
    model = FoundationMILClassifier(**kwargs)
    if cfg["image_size"] != model.vision.config.image_size:
        raise ValueError("Input image size differs from backbone configuration.")
    payload = load_trainable_checkpoint(model, cfg["checkpoint"])
    expected = next(x["locked_threshold"] for x in read_json(ROOT / "model_index.json")["models"]
                    if x["key"] == cfg["modality"])
    if float(payload["threshold"]) != expected:
        raise ValueError("Checkpoint threshold differs from model_index.json.")
    return model, float(payload["threshold"])


def choose_device(name):
    import torch
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; choose --device mps or --device cpu.")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; choose --device cpu.")
    return torch.device(name)


def run_arm(cfg, items, audit, args):
    import numpy as np
    import torch
    import transformers
    from torch.utils.data import DataLoader
    from train_foundation_mil_classifier import evaluate
    device = choose_device(args.device)
    cfg = dict(cfg)
    # Chunking changes only execution batching, not the selected slices/views.
    if device.type == "mps":
        cfg["slice_chunk_size"] = 1
        cfg["eval_batch_size"] = 1
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg["seed"])
    dataset = make_dataset(cfg, items[:1] if args.smoke_test else items)
    loader = DataLoader(dataset, batch_size=1 if args.smoke_test else cfg["eval_batch_size"],
                        shuffle=False, num_workers=args.workers,
                        pin_memory=device.type == "cuda")
    print(f"[{cfg['modality']}] Loading backbone and checkpoint on {device}...", flush=True)
    model, threshold = make_model(cfg)
    model = model.to(device).eval()
    outdir = (resolve_path(args.output_dir) / cfg["modality"] if args.output_dir else cfg["output_dir"])
    metadata = {"device": str(device), "torch_version": torch.__version__,
                "transformers_version": transformers.__version__,
                "precision": "cuda_bfloat16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "float32",
                "checkpoint_sha256": sha256(cfg["checkpoint"]), "threshold": threshold,
                "input_audit": audit}
    private_dir = args.private_output_dir / cfg["modality"] if args.private_output_dir else None
    resolved = {key: str(value) if isinstance(value, Path) else value for key, value in cfg.items()}
    if args.smoke_test:
        import contextlib
        batch = next(iter(loader))
        context = (torch.autocast("cuda", dtype=torch.bfloat16)
                   if device.type == "cuda" and torch.cuda.is_bf16_supported() else contextlib.nullcontext())
        with torch.inference_mode(), context:
            prediction = model(batch["image"].to(device))
        probability = float(torch.sigmoid(prediction["binary_logits"].float()).cpu().reshape(-1)[0])
        if not math.isfinite(probability):
            raise RuntimeError("Non-finite output during inference smoke test.")
        record = {"status": "PASS", "mode": "one_joint_smoke_test", "modality": cfg["modality"],
                  "n_inference_inputs": 1, "finite_output": True,
                  "reference_prediction_comparison_performed": False, **metadata}
        write_json(outdir / "smoke_test.json", record)
        if private_dir:
            write_private_json(private_dir / "smoke_test.json", {
                **record, "joint_id": items[0]["joint_id"], "binary_probability": probability,
                "binary_prediction": int(probability >= threshold)})
            write_private_json(private_dir / "resolved_config.json", resolved)
        print(f"[{cfg['modality']}] Smoke PASS; finite output verified.", flush=True)
    else:
        print(f"[{cfg['modality']}] Evaluating {len(items)} joints...", flush=True)
        def progress_batches():
            for number, batch in enumerate(loader, start=1):
                print(f"[{cfg['modality']}] Batch {number}/{len(loader)}", flush=True)
                yield batch
        primary, three_class, predictions = evaluate(model, progress_batches(), device, threshold)
        reference = read_json(ROOT / "provenance" / cfg["modality"] / "result.json")["primary_binary"]
        result = {"mode": "portable_frozen_checkpoint_evaluation", "modality": cfg["modality"],
                  "primary_binary": primary, "supplementary_three_class": three_class,
                  "historical_reference_auc": reference["auc"],
                  "reference_note": "Historical comparison only; private cohort identity has not been authenticated.",
                  **metadata}
        write_json(outdir / "result.json", result)
        if private_dir:
            write_private_predictions(private_dir, predictions, resolved)
        print(f"[{cfg['modality']}] AUC={primary['auc']:.8f}; frozen checkpoint evaluation.", flush=True)
    del model, loader, dataset
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--config", help="JSON config; relative paths resolve from the package root.")
    selection.add_argument("--arm", choices=("all", *ARMS), default="all")
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Authorized private cohort JSON outside this package. 联系我们获取")
    parser.add_argument("--private-output-dir", type=Path,
                        help="Explicitly enable per-case outputs in this private directory outside the package. Default: aggregate outputs only.")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--data-root", type=Path, required=True,
                        help="External val_images_3k5raw directory; read in place, never copied. Relative to the current working directory.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Verify model hashes, external cohort and PNG counts; no ML dependencies needed.")
    mode.add_argument("--smoke-test", action="store_true", help="Run one joint per selected model; no full-cohort metrics.")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output-dir", help="Optional output root; each arm receives its own subdirectory.")
    args = parser.parse_args()
    if args.workers < 0:
        parser.error("--workers must be non-negative")
    args.manifest = require_external_private_path(args.manifest)
    if not args.manifest.is_file():
        parser.error("Private manifest not found. 联系我们获取")
    if args.private_output_dir:
        args.private_output_dir = require_external_private_path(args.private_output_dir)
        if args.output_dir:
            public_dir = resolve_path(args.output_dir)
            if public_dir == args.private_output_dir or public_dir in args.private_output_dir.parents or args.private_output_dir in public_dir.parents:
                parser.error("Private and aggregate output directories must be separate.")
    paths = ([resolve_path(args.config)] if args.config else
             [ROOT / "configs" / f"{arm}_effective_native_eval_args.json"
              for arm in ARMS if args.arm in ("all", arm)])
    jobs = []
    for path in paths:
        cfg = load_config(path)
        cfg["manifest"] = args.manifest
        external_root = args.data_root.expanduser().resolve()
        cfg["processed_root"] = external_root
        cfg["smri_archive_root"] = None
        cfg["smri_evaluation_archive_root"] = external_root if cfg["modality"] == "smri" else None
        cfg["fusion_archive_root"] = external_root if cfg["modality"] == "fusion" else None
        cfg["evaluation_archive_root"] = external_root if cfg["modality"] in ("cbct", "real_mri") else None
        items, audit = check_inputs(cfg)
        print(f"[{cfg['modality']}] Input check PASS: 74 archive joints, 63 test joints, 60 patients.", flush=True)
        jobs.append((cfg, items, audit))
    if args.check:
        print("Input checks complete. Exact cohort membership and labels require a private source audit.")
        return
    sys.path.insert(0, str(ROOT / "code"))
    for cfg, items, audit in jobs:
        run_arm(cfg, items, audit, args)


if __name__ == "__main__":
    main()
