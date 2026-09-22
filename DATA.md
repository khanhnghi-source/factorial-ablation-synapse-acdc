# Data availability

## What is not here

**No imaging data is redistributed in this repository.** Both benchmarks require
individual registration with their providers, and neither permits redistribution.

| Dataset | How to obtain |
|---|---|
| Synapse Multi-Atlas Labeling (BTCV) | Register at the Synapse platform and accept the challenge terms. |
| ACDC (Automated Cardiac Diagnosis Challenge) | Register on the challenge website and accept its terms. |

Preprocessed `.npz` and `.h5` caches derived from those datasets are likewise not
redistributed, because they are derivative works of the imaging data.

## What is here

| Artefact | Where | Why it can be shared |
|---|---|---|
| Exact partition lists | `SeqAtt_UNet/lists/` and the paper's Appendix | Case identifiers only, no image content. Required by the reviewers so the split can be verified. |
| Per-case metrics for all 70 runs | `results/KetQua_v2.xlsx` | Derived measurements (Dice, HD95) rather than image data. |
| Raw evaluation logs | `results/test_log/` | The text output the metrics are parsed from, released so the aggregation can be checked rather than trusted. |

## Reproducing the preprocessing

The loaders in `SeqAtt_UNet/datasets/` expect the preprocessing convention of the
TransUNet reference implementation. Point `--root_path` at your own preprocessed
cache; the partition lists in `SeqAtt_UNet/lists/` then select exactly the cases
used in the paper.

## A caveat on physical units

The preprocessed volumes do not retain per-case voxel spacing. HD95 is therefore
reported in **voxel** units throughout, and the millimetre column in the paper is
a recomputation under one assumed dataset-level geometry rather than a
per-subject physical distance. Recomputing HD95 with per-case geometry requires
only the original DICOM or NIfTI headers and no retraining.