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
<DATA_ROOT>/labels/forest_nonforest/<same name>.tif        # uint8, values 1..5 (0 = NoData)
<DATA_ROOT>/labels/natural_cultivated/<same name>.tif
```

File naming: `VetSenBR_s{sheet}_a{acquisition}_r{row}_c{col}.tif`. Each sheet id maps to its
official chart code (INOM/MI), biome and acquisition date in `splits/catalog.csv`.

## Layout

```
configs/
  base_vetsenbr.yaml      # training config (overridable: variant, fold, seed, model, loss)
  loss/                   # 6 loss functions (dice_focal, focal, dice, dice_ce, ce, weighted_ce)
splits/
  catalog.csv             # sheet_id -> INOM, MI, biome, acquisition date
  folds_por_carta.csv  folds_por_patch.csv
  csv/<variant>/fold<f>/{train,val,test}.csv
scripts/
  run1_loss_screen.sh   run2_main.sh   aggregate_results.py
data_prep/make_csv_from_folds.py   compute_band_stats.py
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

Model: U-Net / DeepLabV3+ with a ResNet-101 encoder trained from scratch; `classes: 6`
(channel 0 = NoData, ignored → 5 scored classes); AdamW (lr 1e-4), 300 epochs with early
stopping; metrics are macro-averaged (`ignore_index=0`).

## Reproduce the benchmark

```bash
# Loss screening: top-3 losses x 2 architectures x 5 folds x 1 seed = 30 runs
bash scripts/run1_loss_screen.sh <DATA_ROOT>
python scripts/aggregate_results.py --glob "experiments/r1_*"        # pick the best loss

# Main study: 2 architectures x 2 variants x 5 folds x 3 seeds = 60 runs
bash scripts/run2_main.sh <DATA_ROOT> <best_loss>
python scripts/aggregate_results.py --glob "experiments/r2_*" --compare-variants
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
