"""Build only the reviewed public allowlist; never collect private records."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_FILES = ('DATA_ACCESS.json', 'DINOv2_DOWNLOAD.json', 'README.md', 'checkpoints/cbct/best_trainable.pt', 'checkpoints/fusion/target_trainable.pt', 'checkpoints/real_mri/epoch_002_trainable.pt', 'checkpoints/smri/best_trainable.pt', 'code/train_diagnostic_classifier.py', 'code/train_foundation_mil_classifier.py', 'config.json', 'configs/cbct_effective_native_eval_args.json', 'configs/fusion_effective_native_eval_args.json', 'configs/real_mri_effective_native_eval_args.json', 'configs/smri_effective_native_eval_args.json', 'environment/english_conversion_verification.json', 'environment/package_verification.json', 'environment/portable_verification.json', 'environment/privacy_verification.json', 'environment/requirements_core.txt', 'environment/runtime_environment.json', 'environment/shift_description_update.json', 'manifests/classification_locked_binary_626_63_current_patients.json', 'manifests/private_manifest.example.json', 'model.safetensors', 'model_index.json', 'preprocessor_config.json', 'provenance/cbct/evaluation.receipt.json', 'provenance/cbct/invocation.json', 'provenance/cbct/result.json', 'provenance/cbct/verification.json', 'provenance/fusion/invocation.json', 'provenance/fusion/result.json', 'provenance/real_mri/invocation.json', 'provenance/real_mri/result.json', 'provenance/smri/invocation.json', 'provenance/smri/result.json', 'reference/classifier_primary_points.csv', 'reference/classifier_validation.json', 'reproduction/README.md', 'reproduction/build_package.py', 'reproduction/control_source_manifest.json', 'reproduction/launch_package.sh', 'reproduction/package_plan.json', 'run_all.sh', 'run_evaluation.py')


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify():
    manifest = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text())
    rows = manifest["files"]
    if len(rows) != len(PUBLIC_FILES) or {row["path"] for row in rows} != set(PUBLIC_FILES):
        raise ValueError("Inventory differs from the reviewed public allowlist.")
    for row in rows:
        path = ROOT / row["path"]
        if path.is_symlink() or ROOT not in path.resolve().parents:
            raise ValueError("Release members must be regular files inside the package.")
        if not path.is_file() or path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            raise ValueError("Release file changed; review privacy and regenerate the inventory: " + row["path"])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify only (also the default).")
    parser.add_argument("--output", type=Path, help="Explicit ZIP destination outside this package.")
    args = parser.parse_args()
    if args.check and args.output:
        parser.error("Choose --check or --output.")
    rows = verify()
    if args.output:
        target = args.output.expanduser().resolve()
        if target == ROOT or ROOT in target.parents:
            parser.error("The output archive must be outside the package.")
        # Exclusive creation prevents accidental replacement of unrelated files.
        with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as output:
            for name in (*PUBLIC_FILES, "SOURCE_MANIFEST.json"):
                output.write(ROOT / name, "Classifier_Public/" + name)
        print("Public allowlist archive created.")
    else:
        print("PASS: verified %d public files; no archive created." % len(rows))


if __name__ == "__main__":
    main()
