#!/usr/bin/env bash
# =============================================================================
# RUN 1 — loss-function screening (on both architectures, to avoid assuming transfer)
#   top-3 losses x 2 architectures x K=5 folds x 1 seed = 30 runs
#   Fixed variant (forest_nonforest) and seed (42) — this phase only RANKS the loss.
#   Usage:  bash scripts/run1_loss_screen.sh /path/to/dataset
# =============================================================================
set -euo pipefail
DS_ROOT="${1:?usage: run1_loss_screen.sh <DS_ROOT>}"
SEED=42
VARIANT=forest_nonforest
LOSSES=(dice_focal dice weighted_ce)      # top-3 to re-confirm under the new split
ARCHS=(Unet DeepLabV3Plus)

# loss = Hydra group (configs/loss/<name>.yaml): dice_focal, dice, focal, dice_ce, ce, weighted_ce
for lname in "${LOSSES[@]}"; do
  for arch in "${ARCHS[@]}"; do
    for f in 0 1 2 3 4; do
      echo ">>> loss=$lname arch=$arch fold=$f seed=$SEED"
      pytorch-smt --config-dir configs --config-name base_vetsenbr \
        paths.ds_root="$DS_ROOT" variant=$VARIANT fold=$f seed=$SEED \
        model._target_=segmentation_models_pytorch.$arch \
        loss=$lname \
        run_name="r1_${lname}_${arch}_fold${f}_seed${SEED}"
    done
  done
done
echo "RUN 1 done. Aggregate with: python scripts/aggregate_results.py --glob 'experiments/r1_*'"
