#!/usr/bin/env bash
# =============================================================================
# Sonda de LR pra batch=128 (confirmar se 1,6e-4 -- vencedor na mesma sondagem
# feita no dataset E1 do OLD_IP -- ainda eh o melhor aqui no VetSen-BR v2).
# 4 candidatos, 25 epocas cada, fold0/seed42/forest_nonforest, loss=dice_focal
# (fixa, loss ja estabelecida como forte; aqui so testamos LR).
#   Usage:  bash scripts/sweep_lr_bs128.sh /path/to/dataset
# =============================================================================
set -euo pipefail
export PYTHONPATH=.
DS_ROOT="${1:?usage: sweep_lr_bs128.sh <DS_ROOT>}"
LRS=(1e-4 1.6e-4 2.3e-4 4e-4)

for lr in "${LRS[@]}"; do
  echo ">>> sweep_lr bs128 lr=$lr"
  pytorch-smt --config-dir configs --config-name base_vetsenbr \
    paths.ds_root="$DS_ROOT" variant=forest_nonforest fold=0 seed=42 \
    loss=dice_focal hyperparameters.batch_size=128 hyperparameters.epochs=25 \
    optimizer.lr=$lr \
    run_name="sweep_lr_bs128_${lr}"
done
echo "Sweep done. Compare val/MulticlassJaccardIndex and loss/val in wandb (project vetsenbr-segmentation-benchmark, runs sweep_lr_bs128_*)."
