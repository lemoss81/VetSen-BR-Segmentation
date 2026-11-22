#!/usr/bin/env bash
# =============================================================================
# CADEIA FINAL (sequencial; uma etapa que falha NAO impede as seguintes)
#   1) eval no TESTE: r2_* + r1_dice_focal_*   (a celula FNF/seed 42 da Run 2 vem da Run 1)
#   2) agregacao da Run 2 -> experiments/resultado_final_*
#   3) treino da focal pura (Run 1, 10 runs: 2 arq x 5 folds, seed 42)
#   4) eval na VALIDACAO de todas as r1_* (metrica global) + tabela das losses
# Retomavel: eval_test.py pula runs que ja tem o json; run1_loss_screen.sh pula runs concluidas.
# Uso (da raiz do repo, dentro do pixi):
#   pixi run "C:\Program Files\Git\bin\bash.exe" scripts/cadeia_final.sh <DS_ROOT>
# =============================================================================
export PYTHONPATH=.
DS_ROOT="${1:?usage: cadeia_final.sh <DS_ROOT>}"
step() { echo; echo "[$(date '+%F %T')] ===== $* ====="; }

step "1/4 eval no teste (r2_* + r1_dice_focal_*)"
python scripts/eval_test.py --ds-root "$DS_ROOT" --split test --workers 8 --batch-size 64 \
  --runs "experiments/r2_*" "experiments/r1_dice_focal_*" || echo "!!! etapa 1 terminou com erro"

step "2/4 agregacao da Run 2 (teste)"
python scripts/aggregate_results.py --glob "experiments/r2_*" --glob "experiments/r1_dice_focal_*" \
  --loss dice_focal --compare-variants --compare-archs --out experiments/resultado_final \
  | tee experiments/resultado_final_resumo.txt || echo "!!! etapa 2 terminou com erro"

step "3/4 treino da focal pura (LOSSES_OVERRIDE=focal)"
LOSSES_OVERRIDE="focal" bash scripts/run1_loss_screen.sh "$DS_ROOT" || echo "!!! etapa 3 terminou com erro"

step "4/4 eval na validacao de todas as r1_* + tabela das losses"
python scripts/eval_test.py --ds-root "$DS_ROOT" --split val --workers 8 --batch-size 64 \
  --runs "experiments/r1_*" || echo "!!! etapa 4 (eval) terminou com erro"
python scripts/aggregate_results.py --glob "experiments/r1_*" --by-loss --split-file val_check_metrics.json \
  --out experiments/r1_losses_val | tee experiments/r1_losses_val_resumo.txt || echo "!!! etapa 4 (agregacao) terminou com erro"

step "CADEIA CONCLUIDA"
