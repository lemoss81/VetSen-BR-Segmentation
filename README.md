# VetSen-BR — Segmentation benchmark

Treino e avaliação de segmentação semântica no dataset **VetSen-BR** (Sentinel-2, 10 bandas,
256×256), com rótulos derivados de cartas topográficas (ET-EDGV 3.0). Framework:
[`pytorch-segmentation-models-trainer`](https://github.com/dsgoficial/pytorch_segmentation_models_trainer)
(PSMT / Hydra / PyTorch Lightning / `segmentation-models-pytorch`).

**Desenho experimental:** validação cruzada **GroupKFold por carta** (K=5, estratificada por
bioma, split **70/10/20** agrupado — sem vazamento). Duas variantes de rótulo com **5 classes**:
*Forest/Non-forest* (fisionomia) e *Natural/Cultivated* (antropização).

## Layout
```
configs/base_vetsenbr.yaml     # config PSMT (override por run: variant, fold, seed, model, loss)
configs/loss/                  # <-- VOCÊ coloca aqui os 6 configs de loss da ablação
data_prep/
  build_catalog.py             # (já rodado) gera catalog/filename_map/folds em sNNN
  rename_dataset.py            # renomeia os arquivos do dataset p/ sNNN (roda no SSD)
  make_csv_from_folds.py       # gera os CSVs train/val/test 70/10/20 (já rodado)
splits/
  catalog.csv  filename_map.csv            # rastreabilidade (sNNN -> INOM, MI, bioma, data...)
  folds_por_carta.csv  folds_por_patch.csv # folds por carta (sNNN)
  csv/<variante>/fold<f>/{train,val,test}.csv
scripts/
  run1_loss_screen.sh   run2_main.sh   aggregate_results.py
```

## Dados (no SSD — NÃO vão no git)
Estrutura esperada em `<DS_ROOT>`:
```
<DS_ROOT>/images/VetSenBR_sNNN_aK_rRRRR_cCCCC.tif          (10 bandas, uint16)
<DS_ROOT>/labels/forest_nonforest/<mesmo nome>.tif         (uint8, 1..5)
<DS_ROOT>/labels/natural_cultivated/<mesmo nome>.tif
```

## Setup na 5090
```bash
git pull                                   # (ou git clone ... na 1ª vez)
pixi install
# ⚠️ RTX 5090 (Blackwell): se a GPU não for reconhecida, instale PyTorch cu128:
#   pixi run pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## Preparo dos dados (uma vez, no SSD)
```bash
# 1) renomear os arquivos para sNNN (confira em DRY-RUN antes do --apply):
pixi run python data_prep/rename_dataset.py --root <DS_ROOT>
pixi run python data_prep/rename_dataset.py --root <DS_ROOT> --apply
# 2) recomputar a normalização por banda NO v2 (NÃO use a do OLD):
pixi run python compute_band_stats.py --csv splits/csv/forest_nonforest/fold0/train.csv --root-dir <DS_ROOT> --scale-max 65535 --output-json train_band_stats_v2.json
#    -> copie "mean_scaled_list" e "std_scaled_list" do JSON para configs/base_vetsenbr.yaml
#       (blocos Normalize de train e val, campos mean/std)
```
*(os CSVs já estão versionados em `splits/csv/` — não precisa regerar; se quiser, `python data_prep/make_csv_from_folds.py`.)*

## Reprodutibilidade
`pl_trainer.deterministic: true` + seed por run + `pixi.lock` (versões pinadas). **Mesma máquina +
mesma seed = reprodutível.** Bitwise idêntico entre GPUs/drivers/cuDNN diferentes **não** é
garantido (e `16-mixed` adiciona variação) → por isso reportamos média±desvio, não um número único.

## Execução (2 comandos)
```bash
# RUN 1 — triagem de loss: top-3 × 2 arq × 5 folds × 1 semente = 30 runs
bash scripts/run1_loss_screen.sh <DS_ROOT>
python scripts/aggregate_results.py --glob "experiments/r1_*"      # escolhe a loss vencedora

# RUN 2 — principal: 2 arq × 2 variantes × 5 folds × 3 sementes = 60 runs
bash scripts/run2_main.sh <DS_ROOT> <loss_vencedora>
python scripts/aggregate_results.py --glob "experiments/r2_*" --compare-variants
```

## ⚠️ Antes do sweep completo (checklist)
1. **Configs de loss** (`configs/loss/*.yaml`): o repo PSMT traz `WeightedDiceCrossEntropyLoss` e
   outras, mas **Dice+Focal/Focal/Dice-puro vieram da sua configuração** — coloque os 6 aqui e
   ajuste o mapa `LOSS_OVERRIDE` em `scripts/run1_loss_screen.sh` / `run2_main.sh`.
2. **Band stats do v2** copiados para o config (passo acima).
3. **1 DRY-RUN** antes de tudo, p/ validar paths/seed/loss/GPU:
   ```bash
   pytorch-smt --config-dir configs --config-name base_vetsenbr \
     paths.ds_root=<DS_ROOT> variant=forest_nonforest fold=0 seed=42 hyperparameters.epochs=1
   ```
4. Confirme a chave de **seed** do PSMT e o **formato do arquivo de métricas** (ajuste
   `aggregate_results.py` se necessário).
