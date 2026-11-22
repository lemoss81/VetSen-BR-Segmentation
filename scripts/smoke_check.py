# -*- coding: utf-8 -*-
"""
scripts/smoke_check.py
  Confere, SEM treinar, as armadilhas do protocolo novo. Roda em segundos.
  Deve passar antes de disparar qualquer screening.

    pixi run python scripts/smoke_check.py --ds-root <DS_ROOT>
    pixi run python scripts/smoke_check.py --ds-root <DS_ROOT> --arch Segformer --fold 3

  O que NAO da para checar aqui (precisa de um run curto de verdade):
    - a chave do monitor existir em trainer.callback_metrics
    - trainer.estimated_stepping_batches == total_iters
  O script imprime no fim como conferir os dois.
"""
import argparse
import csv
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

FALHAS = []
SELECTED_BANDS = [3, 2, 1, 4, 5, 6, 7, 8, 9, 10]
ENCODER = {"Unet": "resnet101", "DeepLabV3Plus": "resnet101", "Segformer": "mit_b3"}


def check(cond, msg, detalhe=""):
    print("  [%s] %s%s" % ("OK " if cond else "FALHA", msg, ("  -> " + detalhe) if detalhe and not cond else ""))
    if not cond:
        FALHAS.append(msg)
    return cond


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds-root", required=True)
    ap.add_argument("--arch", default="Unet", choices=list(ENCODER))
    ap.add_argument("--variant", default="forest_nonforest")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    print("\n== 1. config do Hydra resolve ==")
    with initialize_config_dir(config_dir=os.path.join(REPO, "configs"), version_base=None):
        cfg = compose(config_name="base_vetsenbr", overrides=[
            # ${hydra:runtime.cwd} so resolve sob o pytorch-smt; aqui informamos o valor
            "paths.repo=%s" % REPO.replace("\\", "/"),
            "paths.ds_root=%s" % args.ds_root.replace("\\", "/"),
            "variant=%s" % args.variant,
            "fold=f%d" % args.fold,         # grupo f0..f4: traz o numero do fold E as band stats dele
            "model._target_=segmentation_models_pytorch.%s" % args.arch,
            "model.encoder_name=%s" % ENCODER[args.arch],
        ])
    c = OmegaConf.to_container(cfg, resolve=True)
    texto = OmegaConf.to_yaml(cfg, resolve=True)
    check(c["hyperparameters"]["classes"] == 5, "hyperparameters.classes == 5")
    check("ignore_index" not in texto, "nenhum ignore_index no config resolvido",
          "a classe 0 e Water; ignore_index=0 apagaria a classe Water inteira")
    check("warmup_epochs" not in texto, "hyperparameters.warmup_epochs ausente",
          "se > 0 o PSMT envelopa o scheduler e cria warmup duplo")
    check(c["train_dataset"]["selected_bands"] == SELECTED_BANDS, "selected_bands na ordem B04,B03,B02,...")
    check(c["val_dataset"]["selected_bands"] == SELECTED_BANDS, "selected_bands igual no val")
    check("csv_derived" in c["train_dataset"]["input_csv_path"], "train le splits/csv_derived")

    print("\n== 2. mapeamento de classes ==")
    from data_prep.make_derived_labels import MASK_CLASS_MAPPING
    py = {int(k): int(v) for k, v in MASK_CLASS_MAPPING[args.variant].items()}
    ya = {int(k): int(v) for k, v in c["mask_class_mapping"].items()}
    check(py == ya, "YAML de variant/%s == MASK_CLASS_MAPPING do Python" % args.variant, str(ya))
    check(sorted(ya) == list(range(1, 11)), "mapeia as 10 subclasses")
    check(sorted(set(ya.values())) == [0, 1, 2, 3, 4], "destino e exatamente 0..4")
    check(c["train_dataset"]["n_classes"] == 5, "train_dataset.n_classes == 5")

    print("\n== 3. estatisticas de banda do fold %d ==" % args.fold)
    with open(os.path.join(REPO, "splits", "band_stats", "fold%d.json" % args.fold), encoding="utf-8") as f:
        bs = json.load(f)
    check(c["fold"] == args.fold, "o grupo fold=%d define fold == %d" % (args.fold, args.fold), str(c["fold"]))
    check(len(c["band_mean"]) == 10 and len(c["band_std"]) == 10, "band_mean/band_std tem 10 valores")
    check(all(abs(a - b) < 1e-12 for a, b in zip(c["band_mean"], bs["mean"])),
          "band_mean do config == splits/band_stats/fold%d.json" % args.fold)
    norm = [t for t in c["train_dataset"]["augmentation_list"] if "Normalize" in str(t.get("_target_", ""))]
    check(len(norm) == 1 and norm[0]["mean"] == c["band_mean"], "Normalize usa band_mean do fold")
    check(bs["selected_bands"] == SELECTED_BANDS, "band_stats gerado na mesma ordem de banda")
    # as band stats de OUTRO fold nao podem estar em uso (o bug que o grupo unico elimina)
    outro = (args.fold + 1) % 5
    with open(os.path.join(REPO, "splits", "band_stats", "fold%d.json" % outro), encoding="utf-8") as f:
        bo = json.load(f)
    check(c["band_mean"] != bo["mean"], "band_mean NAO e o de outro fold (fold%d)" % outro)

    print("\n== 3b. a loss corresponde a este fold e variante ==")
    nome_loss = str(c.get("loss", {}).get("_target_", "")) + " " + json.dumps(c.get("loss_params", {}))
    sel = os.environ.get("SMOKE_LOSS_NAME", "")   # opcional: nome do grupo de loss usado
    if sel.startswith("weighted_ce_"):
        check(sel == "weighted_ce_%s_fold%d" % (args.variant, args.fold),
              "weighted_ce e o do par (variante, fold)", sel)
    else:
        print("  [--] loss sem pesos por fold (%s); nada a conferir" % (sel or "padrao"))
    if "class_weights" in str(c.get("loss", {})):
        cw = c["loss"]["class_weights"]
        check(len(cw) == 5, "class_weights tem 5 entradas (0..4)", str(len(cw)))

    print("\n== 4. cronograma de LR ==")
    sch = c["scheduler_list"][0]
    check(sch["interval"] == "step", "interval == step (por iteracao)")
    check("monitor" not in sch, "sem 'monitor' no item do scheduler", "so ReduceLROnPlateau usa monitor")
    check("PolyWarmupLR" in sch["scheduler"]["_target_"], "scheduler == PolyWarmupLR")
    csv_path = os.path.join(REPO, "splits", "csv_derived", args.variant, "fold%d" % args.fold, "train.csv")
    with open(csv_path, encoding="utf-8") as f:
        n_train = sum(1 for _ in csv.DictReader(f))
    spe = n_train // args.batch
    T = args.epochs * spe
    W = round(0.01 * T)
    print("     n_train=%d  passos/epoca=%d  total_iters=%d  warmup_iters=%d" % (n_train, spe, T, W))

    import torch
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    from custom_sched.poly_warmup import PolyWarmupLR
    s = PolyWarmupLR(opt, total_iters=T, warmup_iters=W, power=1.0)
    curva = []
    for t in range(T):
        curva.append(opt.param_groups[0]["lr"])
        s.step()
    check(abs(curva[0] - 1e-6) < 1e-9, "lr(0) == base * 1e-6", "%g" % curva[0])
    check(all(curva[i] < curva[i + 1] for i in range(W - 1)), "warmup sobe monotonicamente")
    pico = max(curva)
    check(abs(pico - (1.0 - W / T)) < 1e-6, "pico == base * (1 - W/T) = %.4f" % (1.0 - W / T), "%g" % pico)
    check(curva.index(pico) == W, "pico exatamente no fim do warmup", "indice %d" % curva.index(pico))
    check(all(curva[i] > curva[i + 1] for i in range(W, T - 1)), "decai monotonicamente depois do warmup")
    check(curva[-1] < 2.0 / T, "lr do ultimo passo ~ 0", "%g" % curva[-1])

    print("\n== 5. primeira convolucao com 10 canais ==")
    import segmentation_models_pytorch as smp
    cls = {"Unet": smp.Unet, "DeepLabV3Plus": smp.DeepLabV3Plus, "Segformer": smp.Segformer}[args.arch]
    m10 = cls(encoder_name=ENCODER[args.arch], encoder_weights="imagenet", in_channels=10, classes=5, activation=None)
    m3 = cls(encoder_name=ENCODER[args.arch], encoder_weights="imagenet", in_channels=3, classes=5, activation=None)

    def first_conv(mod):
        for name, sub in mod.named_modules():
            if isinstance(sub, torch.nn.Conv2d) and sub.weight.shape[1] in (3, 10) and sub.kernel_size[0] >= 3:
                return name, sub
        return None, None

    n10, c10 = first_conv(m10.encoder)
    n3, c3 = first_conv(m3.encoder)
    print("     modulo: encoder.%s   shape: %s" % (n10, tuple(c10.weight.shape)))
    check(c10.weight.shape[1] == 10, "primeira conv tem 10 canais de entrada")
    check(n10 == n3, "mesmo modulo nas versoes de 3 e 10 canais")
    W10, W3 = c10.weight.detach(), c3.weight.detach()
    iguais = all(torch.allclose(W10[:, i], W3[:, i % 3] * 0.3, atol=1e-6) for i in range(10))
    check(iguais, "canal i == filtro RGB (i % 3) * 3/10 (tile ciclico do smp)")
    razao = float(W10.norm() / W3.norm())
    print("     razao de normas: %.4f  (esperado ~sqrt(0.3)=0.5477, NAO 0.3)" % razao)

    print("\n== 6. um patch real ==")
    import numpy as np
    import rasterio
    from data_prep.make_derived_labels import MASK_CLASS_MAPPING as MM
    lut = np.full(256, 255, dtype=np.uint8)
    for k, v in MM[args.variant].items():
        lut[int(k)] = v
    with open(csv_path, encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))[:200]
    vistos, bandas_ok = set(), True
    for r in linhas:
        with rasterio.open(os.path.join(args.ds_root, r["image"])) as s:
            img = s.read(SELECTED_BANDS)
            bandas_ok &= img.shape[0] == 10
        with rasterio.open(os.path.join(args.ds_root, r["mask"])) as s:
            msk = lut[s.read(1).astype(np.uint8)]
        if (msk == 255).any():
            check(False, "rotulo com valor fora do mask_class_mapping", r["mask"])
            break
        vistos |= set(np.unique(msk).tolist())
    check(bandas_ok, "leitura devolve 10 bandas na ordem pedida")
    check("labels/subclasses/" in linhas[0]["mask"], "a mascara vem de labels/subclasses")
    check(vistos <= {0, 1, 2, 3, 4}, "todos os rotulos em 0..4", str(sorted(vistos)))
    check(vistos == {0, 1, 2, 3, 4}, "as 5 classes aparecem em 200 patches", str(sorted(vistos)))

    print("\n== 7. o SegmentationDataset de verdade ==")
    # Checagem que nenhuma das anteriores cobre: o mask_class_mapping chega ao dataset
    # como DictConfig (interpolacao do OmegaConf). Se o build_mask_class_lut do PSMT nao
    # aceitar isso, o remap falha ou vira no-op e os rotulos iriam como 1..10.
    from hydra.utils import instantiate
    ds = instantiate(cfg.val_dataset, _recursive_=False)
    x, y = None, None
    item = ds[0]
    if isinstance(item, dict):
        x, y = item["image"], item["mask"]
    else:
        x, y = item
    check(tuple(x.shape)[0] == 10, "tensor de imagem com 10 canais", str(tuple(x.shape)))
    vals = set(int(v) for v in y.unique().tolist()) if hasattr(y, "unique") else set(np.unique(y).tolist())
    check(vals <= {0, 1, 2, 3, 4}, "o dataset ja devolve rotulos em 0..4", str(sorted(vals)))
    check(not (vals & {5, 6, 7, 8, 9, 10}), "o remap NAO foi ignorado (nada em 5..10)", str(sorted(vals)))
    check(getattr(ds, "mask_class_lut", None) is not None, "o dataset construiu a LUT de remap")

    print("\n" + "=" * 70)
    if FALHAS:
        print("%d FALHA(S):" % len(FALHAS))
        for f in FALHAS:
            print("  - %s" % f)
        sys.exit(1)
    print("TODAS AS CHECAGENS ESTATICAS PASSARAM.")
    print("""
Falta confirmar num run curto de 2 epocas:

  pixi run pytorch-smt --config-dir configs --config-name base_vetsenbr \\
    paths.ds_root=%s variant=%s fold=f%d \\
    hyperparameters.epochs=2 total_iters=%d warmup_iters=%d run_name=smoke

  1) a chave do monitor existe:  procure "MulticlassJaccardIndex/val" nos escalares do
     TensorBoard em experiments/smoke/ (NAO e "val/MulticlassJaccardIndex").
     Se nao existir, o ModelCheckpoint nunca salva.
  2) trainer.estimated_stepping_batches == total_iters (com 2 epocas, == 2 * %d).
     Se divergir, o accumulate_grad_batches ou o drop_last mudaram a conta.
  3) a curva lr-poly_warmup sobe por %d passos e depois cai.
""" % (args.ds_root, args.variant, args.fold, 2 * spe, round(0.01 * 2 * spe) or 1, spe, max(1, round(0.01 * 2 * spe))))


if __name__ == "__main__":
    main()
