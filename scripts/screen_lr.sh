#!/usr/bin/env bash
# =============================================================================
# SCREENING 1 - LR x loss, por arquitetura, nos DOIS esquemas, pontuado na VALIDACAO.
#
#   3 arquiteturas x 3 LR x 4 losses x 2 esquemas x 5 folds x 1 semente = 360 runs
#
# Simetrico por construcao: a mesma grade roda em forest_nonforest e natural_cultivated,
# e nada e pontuado no teste. Isso responde a critica de "screening so no FNF".
#
#   bash scripts/screen_lr.sh <DS_ROOT>
#
# Grade reduzida (ex.: so o piloto, ou uma fatia por vez):
#   ARCHS=Unet LOSSES=dice_focal FOLDS=0 VARIANTS=forest_nonforest bash scripts/screen_lr.sh <DS_ROOT>
#
# Desenho fatorado (90 + 90 em vez de 360): rode primeiro so com LOSSES=dice_focal
# para escolher o LR por arquitetura, depois fixe LRS_<arch> no LR vencedor e rode as
# 4 losses. Mesma simetria entre esquemas, metade do custo.
#
# Retomavel: pula run que ja tem final_metrics.json.
# =============================================================================
set -euo pipefail
export PYTHONPATH=.        # custom_losses.* e custom_sched.* resolvidos pelo Hydra
export WANDB_MODE=offline  # sem isso o wandb ABRE UM PROMPT de login a cada run e trava o lote.
                           # 'offline' grava em experiments/wandb/ e pode ser sincronizado depois
                           # (wandb sync); use 'disabled' para nao gravar nada.

DS_ROOT="${1:?uso: screen_lr.sh <DS_ROOT>}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-150}"
BATCH="${BATCH:-128}"              # batch EFETIVO (micro-batch x accumulate)
read -ra ARCHS    <<< "${ARCHS:-Unet DeepLabV3Plus Segformer}"
read -ra LOSSES   <<< "${LOSSES:-dice_focal dice focal weighted_ce}"
read -ra VARIANTS <<< "${VARIANTS:-forest_nonforest natural_cultivated}"
read -ra FOLDS    <<< "${FOLDS:-0 1 2 3 4}"

# LR por arquitetura (o transformer pede faixa mais baixa; 6e-5 e o do artigo do SegFormer)
LRS_Unet="${LRS_Unet:-1e-4 1.6e-4 2.5e-4}"
LRS_DeepLabV3Plus="${LRS_DeepLabV3Plus:-1e-4 1.6e-4 2.5e-4}"
LRS_Segformer="${LRS_Segformer:-6e-5 1e-4 1.6e-4}"

# encoder e acumulacao de gradiente por arquitetura.
# ACCUM > 1 mantem o batch EFETIVO e o numero de optimizer steps (logo o T) -- so reduz VRAM.
ENC_Unet="${ENC_Unet:-resnet101}";                 ACC_Unet="${ACC_Unet:-1}"
ENC_DeepLabV3Plus="${ENC_DeepLabV3Plus:-resnet101}"; ACC_DeepLabV3Plus="${ACC_DeepLabV3Plus:-1}"
ENC_Segformer="${ENC_Segformer:-mit_b3}";          ACC_Segformer="${ACC_Segformer:-2}"

var() { eval "printf '%s' \"\${$1}\""; }

echo "=== SCREENING 1 (validacao) ==="
echo "  arquiteturas: ${ARCHS[*]}"
echo "  losses:       ${LOSSES[*]}"
echo "  esquemas:     ${VARIANTS[*]}"
echo "  folds:        ${FOLDS[*]}   seed: $SEED   epochs: $EPOCHS   batch efetivo: $BATCH"
echo

total=0; feitas=0
for arch in "${ARCHS[@]}"; do
  read -ra LRS <<< "$(var "LRS_${arch}")"
  ENC="$(var "ENC_${arch}")"
  ACCUM="$(var "ACC_${arch}")"
  MICRO=$(( BATCH / ACCUM ))
  for lr in "${LRS[@]}"; do
    for lname in "${LOSSES[@]}"; do
      for v in "${VARIANTS[@]}"; do
        for f in "${FOLDS[@]}"; do
          total=$(( total + 1 ))

          # ---- cronograma, calculado do train.csv DESTE fold (drop_last: true) ----
          CSV="splits/csv_derived/${v}/fold${f}/train.csv"
          N=$(( $(wc -l < "$CSV") - 1 ))
          SPE=$(( N / BATCH ))                 # optimizer steps por epoca
          T=$(( EPOCHS * SPE ))
          W=$(( (T + 50) / 100 ))              # 1% de T, arredondado
          [ "$W" -lt 1 ] && W=1

          # weighted_ce tem pesos por fold e por variante
          LOSS_ARG="$lname"
          if [ "$lname" = "weighted_ce" ]; then LOSS_ARG="weighted_ce_${v}_fold${f}"; fi

          RN="s1_${arch}_lr${lr}_${lname}_${v}_fold${f}_seed${SEED}"
          if compgen -G "experiments/$RN/version_*/final_metrics.json" > /dev/null; then
            echo ">>> ja concluida, pulando: $RN"
            feitas=$(( feitas + 1 )); continue
          fi
          echo ">>> $RN  | enc=$ENC micro=$MICRO accum=$ACCUM N=$N spe=$SPE T=$T warmup=$W"

          pytorch-smt --config-dir configs --config-name base_vetsenbr \
            paths.ds_root="$DS_ROOT" \
            variant="$v" fold="f${f}" seed="$SEED" \
            loss="$LOSS_ARG" \
            model._target_="segmentation_models_pytorch.${arch}" \
            model.encoder_name="$ENC" \
            optimizer.lr="$lr" \
            hyperparameters.epochs="$EPOCHS" \
            hyperparameters.batch_size="$MICRO" \
            pl_trainer.accumulate_grad_batches="$ACCUM" \
            total_iters="$T" warmup_iters="$W" \
            run_name="$RN"
          feitas=$(( feitas + 1 ))
        done
      done
    done
  done
done
echo
echo "SCREENING 1 concluido: $feitas/$total run(s)."
echo "Avaliar na VALIDACAO e agregar (NUNCA no teste nesta fase):"
echo "  python scripts/eval_test.py --ds-root <DS_ROOT> --split val --runs 'experiments/s1_*'"
echo "  python scripts/aggregate_results.py --glob 'experiments/s1_*' --by-loss --split-file val_check_metrics.json"
