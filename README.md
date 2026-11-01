# Frozen classifier inference package

This package contains four classifier checkpoints, the shared DINOv2 backbone,
inference code, evaluation configurations, and recorded aggregate results.
It supports inference with existing weights; training and model-selection
history are not included. Package checks verify file consistency and software
operation, not study validity. No model was retrained and no patient-cohort
results were recomputed during this packaging update.

## Private data

患者名单、训练／测试／排除病例清单、患者编号映射及逐病例预测：**联系我们获取**。
Patient lists, identity mappings, images and individual predictions require
authorized access and are not distributed in this package. Keep private inputs
and individual outputs outside this folder.

## Evaluation

Download the checkpoint from
[Google Drive Link](https://drive.google.com/file/d/1u5CBztcBR5WfwAcbtgsbX4OUyDJC9_sH/view?usp=sharing)

Install `environment/requirements_core.txt` in your evaluation environment.
The DINOv2 backbone is included locally.

```bash
python3 run_evaluation.py --manifest /path/to/private_manifest.json --data-root /path/to/private_images --check
python3 run_evaluation.py --manifest /path/to/private_manifest.json --data-root /path/to/private_images --smoke-test --arm cbct
python3 run_evaluation.py --manifest /path/to/private_manifest.json --data-root /path/to/private_images --output-dir /path/to/aggregate_results
```

The private manifest must contain an `items` array with `joint_id`, `patient_id`,
`side` (`L` or `R`), integer `label` (0 = normal, 1 = anterior displacement), and
`split`. Evaluation uses `split: test` and requires 63 joints, 60 patient-ID
values, and labels 28/35. The image archive must contain 74 joint directories
with 1,462 PNGs per required view, 15–25 PNGs per joint. Counts and format checks
do not authenticate patient identities, cohort membership or labels; verify
these against the authorized source records. The schema example contains no
patient records and cannot be evaluated.

Inputs are `condition` for CBCT, `shift` for sMRI, `condition+shift` for fusion,
and `gt` for real MRI. The `shift` images have undergone brightness correction.
Thresholds are loaded from the supplied checkpoints and are not fitted during
evaluation. The default output contains aggregate metrics. To export individual
results, add `--private-output-dir /path/to/private_results`; keep these private.
The smoke test verifies finite output for one input, not agreement of a full
cohort with the recorded results.

## Packaging

```bash
python3 reproduction/build_package.py --check
python3 reproduction/build_package.py --output /path/outside/this/folder/classifier_public.zip
```

The builder verifies an explicit file allowlist against `SOURCE_MANIFEST.json`.
Private data, runtime outputs, caches and unlisted files are excluded. See
`environment/package_verification.json` for the scope of this update's checks.
