#!/usr/bin/env bash
# =============================================================================
# RUN 2 — main experiment (with the winning loss from RUN 1)
#   2 architectures x 2 variants x K=5 folds x 3 seeds = 60 runs
#   (forest_nonforest + seed 42 e pulado: ja foi executado no RUN 1 com a mesma loss/config)
#   -> mean+-std over 15 (fold x seed) per cell; Wilcoxon between variants.
#   Usage:  bash scripts/run2_main.sh /path/to/dataset dice_focal
# Retomavel: runs com final_metrics.json sao puladas; se uma run cair, guarda a tentativa
# em experiments/_falhou_* e repete 1x com num_workers=8.
# =============================================================================
set -euo pipefail
export PYTHONPATH=.    # so Hydra can resolve custom_losses.* (vendored WeightedDiceCrossEntropyLoss)
DS_ROOT="${1:?usage: run2_main.sh <DS_ROOT> <winning_loss>}"
WIN_LOSS="${2:?provide the winning loss from RUN 1 (e.g. dice_focal)}"
ARCHS=(Unet DeepLabV3Plus)
VARIANTS=(forest_nonforest natural_cultivated)
SEEDS=(42 43 44)

for arch in "${ARCHS[@]}"; do
  for v in "${VARIANTS[@]}"; do
    for f in 0 1 2 3 4; do
      for s in "${SEEDS[@]}"; do

        if [ "$v" = "forest_nonforest" ] && [ "$s" = 42 ]; then
            echo "Pulando forest_nonforest com seed 42 (já executado no RUN 1)"
            continue
        fi

        # weighted_ce has per-variant class weights: use the NC config on natural_cultivated
        LOSS_ARG=$WIN_LOSS
        if [ "$WIN_LOSS" = "weighted_ce" ] && [ "$v" = "natural_cultivated" ]; then LOSS_ARG=weighted_ce_nc; fi

        RN="r2_${arch}_${v}_fold${f}_seed${s}"
        if compgen -G "experiments/$RN/version_*/final_metrics.json" > /dev/null; then
          echo ">>> JA CONCLUIDA, pulando: $RN"
          continue
        fi
        echo ">>> arch=$arch variant=$v fold=$f seed=$s loss=$LOSS_ARG"
        if ! pytorch-smt --config-dir configs --config-name base_vetsenbr \
          paths.ds_root="$DS_ROOT" variant=$v fold=$f seed=$s \
          model._target_=segmentation_models_pytorch.$arch \
          loss=$LOSS_ARG \
          run_name="$RN"; then
          echo "!!! $RN falhou -- guardando a tentativa e repetindo 1x com num_workers=8"
          mv "experiments/$RN" "experiments/_falhou_${RN}_$(date +%s)" 2>/dev/null || true
          pytorch-smt --config-dir configs --config-name base_vetsenbr \
            paths.ds_root="$DS_ROOT" variant=$v fold=$f seed=$s \
            model._target_=segmentation_models_pytorch.$arch \
            loss=$LOSS_ARG \
            train_dataset.data_loader.num_workers=8 val_dataset.data_loader.num_workers=8 \
            run_name="$RN"
        fi
      done
    done
  done
done
echo "RUN 2 done. Aggregate with: python scripts/aggregate_results.py --glob 'experiments/r2_*' --compare-variants"
