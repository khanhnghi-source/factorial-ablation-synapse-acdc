# Third-party code, weights and data

`LICENSE` (Apache-2.0) covers the source code in this repository, both the parts
inherited from TransUNet and the parts written for this study. It does not
override the terms below, which continue to apply to the components they name.

## Source code

| Component | Upstream | Licence | Notes |
|---|---|---|---|
| Model, trainer and evaluation scaffolding | [TransUNet](https://github.com/Beckschen/TransUNet) (Chen et al., 2021) | Apache-2.0 | This repository is a derivative work. Sixteen files are derived from it and each carries a modification notice; they are listed in `NOTICE`. Upstream ships no `NOTICE` file of its own, so Section 4(d) does not apply, and its source files carry no copyright headers, so Section 4(c) leaves nothing to preserve. |
| ViT configuration and weight-loading | [Google Research `vision_transformer`](https://github.com/google-research/vision_transformer) | Apache-2.0 | Reached through TransUNet; the same obligations are discharged by the same `LICENSE` and `NOTICE`. |

## Pretrained weights

| Weights | Source | Licence | Notes |
|---|---|---|---|
| R50+ViT-B_16, ImageNet-21k | Google Research | Apache-2.0 for the code; the ImageNet-21k images are **not** Apache-2.0 and remain subject to the ImageNet terms of access | Not redistributed here. Download from the upstream release; see `DATA.md`. |

## Libraries

PyTorch, medpy, SimpleITK, scikit-image, scipy, timm, einops, fvcore and the
remaining dependencies in `requirements.txt` are used unmodified under their own
licences.

## Datasets

Not redistributed. See `DATA.md`.