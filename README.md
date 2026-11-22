# VetSen-BR — Segmentation Benchmark

Training and evaluation code for **VetSen-BR**, a semantic-segmentation benchmark that pairs
Sentinel-2 imagery (10 bands, 256×256 at 10 m) with reference labels derived from Brazilian Army
topographic charts structured under the ET-EDGV 3.0 specification.

The dataset is released in two labeling variants, each with **5 classes** (Water, Built-up,
Bare Soil + two vegetation classes):

- **Forest / Non-forest** — vegetation grouped by physiognomy (tree-form vs. shrub/herbaceous).
- **Natural / Cultivated** — vegetation grouped by anthropization (native vs. planted/managed).

**Evaluation protocol:** 5-fold **grouped cross-validation by chart** (no patch from a chart is
split across folds), stratified by biome, with a **70/10/20** train/val/test partition. The exact
fold and split assignments are shipped in `splits/` for full reproducibility.

Framework: [`pytorch-segmentation-models-trainer`](https://github.com/dsgoficial/pytorch_segmentation_models_trainer)
(Hydra · PyTorch Lightning · [`segmentation-models-pytorch`](https://github.com/qubvel-org/segmentation_models.pytorch)).

## Dataset

Download the dataset and place it on fast local storage. Expected layout (`<DATA_ROOT>`):

```
<DATA_ROOT>/images/VetSenBR_s001_a1_r0256_c0512.tif        # 10-band stack, uint16
<DATA_ROOT>/labels/subclasses/<same name>.tif              # uint8, values 1..10 — read by training
<DATA_ROOT>/labels/forest_nonforest/<same name>.tif        # uint8, values 1..5
<DATA_ROOT>/labels/natural_cultivated/<same name>.tif
```

**Labels have no NoData.** Every one of the 610,795,520 pixels carries a real class, in all
three variants — the tiling step discards any patch with an unlabeled pixel, so `0` never
occurs. Training therefore uses **5 classes (0..4) with no `ignore_index`**; class `0` is
Water.

**Training reads `labels/subclasses/`,** not the aggregated folders, and derives each
variant on the fly with `mask_class_mapping` (see `splits/csv_derived/README.md`). This is
exact for `natural_cultivated` (0 differing pixels) and corrects 8,275 pixels in 54 patches
of `forest_nonforest`, where a rasterisation-priority bug put Silviculture in Non-forest.

File naming: `VetSenBR_s{sheet}_a{acquisition}_r{row}_c{col}.tif`. Each sheet id maps to its
official chart code (INOM/MI), biome and acquisition date in `splits/catalog.csv`.

## Layout

```
configs/
  base_vetsenbr.yaml      # training config (overridable: variant, fold, seed, model, loss)
  variant/                # per-variant mask_class_mapping (subclasses 1..10 -> 0..4)
  fold/<f>.yaml           # fold number + that fold's band mean/std (one knob, cannot diverge)
  loss/                   # dice_focal, focal, dice, dice_ce, ce + weighted_ce_<variant>_fold<f>
custom_sched/poly_warmup.py   # linear warmup + poly decay, per iteration
splits/
  catalog.csv             # sheet_id -> INOM, MI, biome, acquisition date
  folds_por_carta.csv  folds_por_patch.csv
  csv/<variant>/fold<f>/{train,val,test}.csv          # as published on Zenodo
  csv_derived/<variant>/fold<f>/{train,val,test}.csv  # same splits, masks from labels/subclasses
  band_stats/fold<f>.json # per-band mean/std from that fold's training split
scripts/
  smoke_check.py   screen_lr.sh   eval_test.py   aggregate_results.py
  _v1_obsoleto/           # v1.0 protocol (classes 6) — do not run
data_prep/make_csv_from_folds.py   make_derived_labels.py   compute_band_stats.py
```

## Installation

```bash
git clone https://github.com/lemoss81/VetSen-BR-Segmentation.git
cd VetSen-BR-Segmentation
pixi install
```
> **NVIDIA Blackwell (e.g. RTX 5090):** if the GPU is not detected, install a CUDA 12.8 build of PyTorch:
> `pixi run pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`

## Quick start — train one model

```bash
pytorch-smt --config-dir configs --config-name base_vetsenbr \
  paths.ds_root=<DATA_ROOT> variant=forest_nonforest fold=0 seed=42 loss=dice_focal
```
Override any of: `variant` (`forest_nonforest` | `natural_cultivated`), `fold` (`0..4`),
`seed`, `loss` (`dice_focal` | `focal` | `dice` | `dice_ce` | `ce` | `weighted_ce`),
`model._target_` (`segmentation_models_pytorch.Unet` | `...DeepLabV3Plus`).

Models: U-Net and DeepLabV3+ on a ResNet-101 encoder, and SegFormer on a MiT-B3 encoder —
all three **ImageNet-pretrained**. Bands are reordered to `[3,2,1,4,…]` (B04,B03,B02,…) so
the RGB-derived filters land on the visible bands; inputs are per-band z-scored with
statistics computed **per fold, from that fold's training split only**.

Optimiser AdamW; **linear warmup (1% of total steps) then polynomial decay to zero, applied
per iteration**; a fixed budget of 150 epochs with **no early stopping**, since stopping
early would truncate the schedule. The checkpoint is selected by validation mIoU
(`MulticlassJaccardIndex/val`, `mode: max`), not by the loss — the loss is on a different
scale for each candidate, so using it would bias the loss comparison. Metrics are
macro-averaged over the 5 classes, with no `ignore_index`.

## Reproduce the benchmark

```bash
# 0. Smoke check — static assertions, seconds, no training. Run this first.
python scripts/smoke_check.py --ds-root <DATA_ROOT>

# 1. Screening: 3 architectures x 3 LRs x 4 losses x 2 variants x 5 folds x 1 seed = 360 runs
#    Symmetric across BOTH variants and scored on VALIDATION only.
bash scripts/screen_lr.sh <DATA_ROOT>
python scripts/eval_test.py --ds-root <DATA_ROOT> --split val --runs "experiments/s1_*"
python scripts/aggregate_results.py --glob "experiments/s1_*" --by-loss --by-lr
```
Run inside the environment (`pixi run bash scripts/...`; on Windows use Git Bash).

## Reproducibility

Deterministic training (`pl_trainer.deterministic: true`, fixed seeds) with the environment pinned
by `pixi.lock`. Results are reproducible on the same machine; bitwise-identical results across
different GPUs/drivers/cuDNN are not guaranteed (and `16-mixed` precision adds minor variation),
so results are reported as mean ± standard deviation over folds and seeds.

## Citation

If you use VetSen-BR, please cite:

```bibtex
@misc{vetsenbr,
title = {VetSen-BR: a Sentinel-2 semantic-segmentation benchmark for Brazil},
author = {Mateus L. de Abrantes and André L. Favalessa and Maurício C. M. de Paulo and Felipe Ferrari and Raphael L. F. Greco and Philipe Borba and Daniel R. dos Santos},
year = {2026},
note = {https://www.kaggle.com/datasets/datasetbr1/vetsen-br}
}
```
