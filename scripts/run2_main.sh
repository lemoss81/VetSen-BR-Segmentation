#!/usr/bin/env bash
# =============================================================================
# RUN 2 — experimento principal (com a loss vencedora da RUN 1)
#   2 arquiteturas × 2 variantes × K=5 folds × 3 sementes = 60 runs
#   -> média±desvio sobre 15 (fold×seed) por célula; Wilcoxon entre variantes.
#   Uso:  bash scripts/run2_main.sh /caminho/no/SSD/dataset dice_focal
# =============================================================================
set -euo pipefail
DS_ROOT="${1:?uso: run2_main.sh <DS_ROOT> <loss_vencedora>}"
WIN_LOSS="${2:?informe a loss vencedora da RUN 1 (ex.: dice_focal)}"
ARCHS=(Unet DeepLabV3Plus)
VARIANTS=(forest_nonforest natural_cultivated)
SEEDS=(42 43 44)

declare -A LOSS_OVERRIDE
LOSS_OVERRIDE[dice_focal]="loss=dice_focal"
LOSS_OVERRIDE[dice]="loss=dice"
LOSS_OVERRIDE[weighted_ce]="loss=weighted_ce"

for arch in "${ARCHS[@]}"; do
  for v in "${VARIANTS[@]}"; do
    for f in 0 1 2 3 4; do
      for s in "${SEEDS[@]}"; do
        echo ">>> arch=$arch variant=$v fold=$f seed=$s loss=$WIN_LOSS"
        pytorch-smt --config-dir configs --config-name base_vetsenbr \
          paths.ds_root="$DS_ROOT" variant=$v fold=$f seed=$s \
          model._target_=segmentation_models_pytorch.$arch \
          ${LOSS_OVERRIDE[$WIN_LOSS]} \
          run_name="r2_${arch}_${v}_fold${f}_seed${s}"
      done
    done
  done
done
echo "RUN 2 concluída. Agregue com: python scripts/aggregate_results.py --glob 'experiments/r2_*' --compare-variants"
