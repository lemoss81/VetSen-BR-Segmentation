#!/usr/bin/env bash
# =============================================================================
# RUN 1 — loss-function screening (on both architectures, to avoid assuming transfer)
#   top-3 losses x 2 architectures x K=5 folds x 1 seed = 30 runs
#   Fixed variant (forest_nonforest) and seed (42) — this phase only RANKS the loss.
#   Usage:  bash scripts/run1_loss_screen.sh /path/to/dataset
# =============================================================================
set -euo pipefail
export PYTHONPATH=.    # so Hydra can resolve custom_losses.* (vendored WeightedDiceCrossEntropyLoss)
DS_ROOT="${1:?usage: run1_loss_screen.sh <DS_ROOT>}"
SEED=42
VARIANT=forest_nonforest
# top-3 to re-confirm under the new split; LOSSES_OVERRIDE="focal" roda so outras losses (ex.: focal pura)
read -ra LOSSES <<< "${LOSSES_OVERRIDE:-dice_focal dice weighted_ce}"
ARCHS=(Unet DeepLabV3Plus)

# loss = Hydra group (configs/loss/<name>.yaml): dice_focal, dice, focal, dice_ce, ce, weighted_ce
for lname in "${LOSSES[@]}"; do
  for arch in "${ARCHS[@]}"; do
    for f in 0 1 2 3 4; do
      RN="r1_${lname}_${arch}_fold${f}_seed${SEED}"
      # Retomada: pula runs que ja terminaram (final_metrics.json existe), para poder
      # reiniciar o script sem refazer o que esta pronto.
      if compgen -G "experiments/$RN/version_*/final_metrics.json" > /dev/null; then
        echo ">>> JA CONCLUIDA, pulando: $RN"
        continue
      fi
      echo ">>> loss=$lname arch=$arch fold=$f seed=$SEED"
      if ! pytorch-smt --config-dir configs --config-name base_vetsenbr \
        paths.ds_root="$DS_ROOT" variant=$VARIANT fold=$f seed=$SEED \
        model._target_=segmentation_models_pytorch.$arch \
        loss=$lname \
        run_name="$RN"; then
        # Rodada sem supervisao: no Windows, batch grande + muitos workers persistentes pode
        # derrubar o treino com 'Couldn't open shared file mapping ... error code 1455'
        # (visto no OLD_IP por volta das epocas 22-29). Guarda a tentativa que falhou fora do
        # padrao r1_* e repete UMA vez com num_workers=8 (nao altera o protocolo estatistico);
        # se a repeticao tambem falhar, o 'set -e' interrompe a cadeia.
        echo "!!! $RN falhou -- guardando a tentativa e repetindo 1x com num_workers=8"
        mv "experiments/$RN" "experiments/_falhou_${RN}_$(date +%s)" 2>/dev/null || true
        pytorch-smt --config-dir configs --config-name base_vetsenbr \
          paths.ds_root="$DS_ROOT" variant=$VARIANT fold=$f seed=$SEED \
          model._target_=segmentation_models_pytorch.$arch \
          loss=$lname \
          train_dataset.data_loader.num_workers=8 val_dataset.data_loader.num_workers=8 \
          run_name="$RN"
      fi
    done
  done
done
echo "RUN 1 done. Aggregate with: python scripts/aggregate_results.py --glob 'experiments/r1_*'"
