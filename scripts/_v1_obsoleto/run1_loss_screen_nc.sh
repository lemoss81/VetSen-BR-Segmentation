#!/usr/bin/env bash
# =============================================================================
# RUN 1 (NC) — loss-function screening on NATURAL/CULTIVATED.
#   Mirrors scripts/run1_loss_screen.sh but on the natural_cultivated scheme, to
#   confirm that the loss choice transfers across schemes (addresses the review:
#   "screening done only on Forest/Non-forest and applied to Natural/Cultivated").
#   4 losses x 2 architectures x K=5 folds x 1 seed (42) = 40 runs, scored on VALIDATION.
#   Run names: r1_<loss>_<arch>_natural_cultivated_fold<f>_seed42
#     -> parsed by eval_test.py (variant=natural_cultivated) and does NOT collide
#        with the existing FNF r1_* runs (which omit the variant in the name).
#   weighted_ce uses the per-variant config weighted_ce_nc, but keeps the label
#   "weighted_ce" in the run name so aggregate_results groups it across schemes.
#   Usage:  bash scripts/run1_loss_screen_nc.sh /path/to/dataset
# =============================================================================
set -euo pipefail
export PYTHONPATH=.    # so Hydra can resolve custom_losses.* (vendored WeightedDiceCrossEntropyLoss)
DS_ROOT="${1:?usage: run1_loss_screen_nc.sh <DS_ROOT>}"
SEED=42
VARIANT=natural_cultivated
read -ra LOSSES <<< "${LOSSES_OVERRIDE:-dice_focal dice focal weighted_ce}"
ARCHS=(Unet DeepLabV3Plus)

for lname in "${LOSSES[@]}"; do
  for arch in "${ARCHS[@]}"; do
    for f in 0 1 2 3 4; do
      # weighted_ce has per-variant class weights -> use the NC config, keep label in name
      LOSS_ARG=$lname
      if [ "$lname" = "weighted_ce" ]; then LOSS_ARG=weighted_ce_nc; fi
      RN="r1_${lname}_${arch}_natural_cultivated_fold${f}_seed${SEED}"
      if compgen -G "experiments/$RN/version_*/final_metrics.json" > /dev/null; then
        echo ">>> JA CONCLUIDA, pulando: $RN"
        continue
      fi
      echo ">>> loss=$lname arch=$arch fold=$f seed=$SEED variant=$VARIANT"
      if ! pytorch-smt --config-dir configs --config-name base_vetsenbr \
        paths.ds_root="$DS_ROOT" variant=$VARIANT fold=$f seed=$SEED \
        model._target_=segmentation_models_pytorch.$arch \
        loss=$LOSS_ARG \
        run_name="$RN"; then
        echo "!!! $RN falhou -- guardando a tentativa e repetindo 1x com num_workers=8"
        mv "experiments/$RN" "experiments/_falhou_${RN}_$(date +%s)" 2>/dev/null || true
        pytorch-smt --config-dir configs --config-name base_vetsenbr \
          paths.ds_root="$DS_ROOT" variant=$VARIANT fold=$f seed=$SEED \
          model._target_=segmentation_models_pytorch.$arch \
          loss=$LOSS_ARG \
          train_dataset.data_loader.num_workers=8 val_dataset.data_loader.num_workers=8 \
          run_name="$RN"
      fi
    done
  done
done
echo "RUN 1 (NC) done. Evaluate on VALIDATION and aggregate by loss:"
echo "  python scripts/eval_test.py --ds-root <DS_ROOT> --split val --runs 'experiments/r1_*natural_cultivated*'"
echo "  python scripts/aggregate_results.py --glob 'experiments/r1_*natural_cultivated*' --by-loss --split-file val_check_metrics.json"
