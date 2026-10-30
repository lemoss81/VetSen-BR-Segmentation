#!/usr/bin/env bash
# =============================================================================
# RUN 1 — triagem de função de perda (nas 2 arquiteturas, p/ não assumir transferência)
#   top-3 losses × 2 arquiteturas × K=5 folds × 1 semente = 30 runs
#   Variante fixa (forest_nonforest) e semente fixa (42) — aqui só RANQUEIA a loss.
#   Uso:  bash scripts/run1_loss_screen.sh /caminho/no/SSD/dataset
# =============================================================================
set -euo pipefail
DS_ROOT="${1:?uso: run1_loss_screen.sh <DS_ROOT>}"
SEED=42
VARIANT=forest_nonforest
LOSSES=(dice_focal dice weighted_ce)      # top-3 do histórico (reconfirmar sob o split novo)
ARCHS=(Unet DeepLabV3Plus)

# loss = grupo Hydra (configs/loss/<nome>.yaml): dice_focal, dice, focal, dice_ce, ce, weighted_ce
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
echo "RUN 1 concluída. Agregue com: python scripts/aggregate_results.py --glob 'experiments/r1_*'"
