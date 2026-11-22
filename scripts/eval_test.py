# -*- coding: utf-8 -*-
"""
scripts/eval_test.py
  Avalia o MELHOR checkpoint (maior MulticlassJaccardIndex/val) de cada run no split
  pedido e grava <run>/test_metrics.json -- entrada do scripts/aggregate_results.py.

  ATUALIZADO para o protocolo de classes 5 (ver configs/base_vetsenbr.yaml):
    - 5 classes, 0..4, SEM ignore_index. A classe 0 e Water, nao NoData: o dataset
      nao tem nenhum pixel NoData (medido: 0 em 610.795.520).
    - as mascaras vem de labels/subclasses e sao remapeadas pelo MASK_CLASS_MAPPING
      da variante (mesma fonte usada por data_prep/make_derived_labels.py).
    - bandas na ordem [3,2,1,4,5,6,7,8,9,10] = B04,B03,B02,B08,B05,B06,B07,B8A,B11,B12.
    - mean/std do fold da run, lidos de splits/band_stats/fold<f>.json.

  Uso:
    python scripts/eval_test.py --ds-root <DS_ROOT> --split val --runs "experiments/s1_*"
    python scripts/eval_test.py --ds-root <DS_ROOT> --split test --runs "experiments/s2_*"

  O split 'test' e RECUSADO para runs de screening (s1_*): a fase de screening e
  pontuada so na validacao. Use --force-test para sobrepor (nao faca isso).
"""
import argparse
import datetime
import glob
import json
import os
import re
import sys
import time

import albumentations as A
import numpy as np
import pandas as pd
import rasterio
import segmentation_models_pytorch as smp
import torch
import torchmetrics as tm
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from data_prep.make_derived_labels import MASK_CLASS_MAPPING, NOMES  # fonte unica do remap

SELECTED_BANDS = [3, 2, 1, 4, 5, 6, 7, 8, 9, 10]
N_CLASSES = 5

# s1_<arch>[_lr<lr>][_<loss>]_<variant>_fold<f>_seed<s>
RUN_RE = re.compile(
    r"^(?P<phase>s\d)_(?P<arch>Unet|DeepLabV3Plus|Segformer)"
    r"(?:_lr(?P<lr>[0-9.eE+-]+))?"
    r"(?:_(?P<loss>[a-z0-9_]+?))?"
    r"_(?P<variant>forest_nonforest|natural_cultivated)"
    r"_fold(?P<fold>\d+)_seed(?P<seed>\d+)$"
)
ENCODER = {"Unet": "resnet101", "DeepLabV3Plus": "resnet101", "Segformer": "mit_b3"}


def band_stats(fold):
    p = os.path.join(REPO, "splits", "band_stats", "fold%s.json" % fold)
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    if d["selected_bands"] != SELECTED_BANDS:
        raise ValueError("%s tem selected_bands %s, esperado %s" % (p, d["selected_bands"], SELECTED_BANDS))
    return d["mean"], d["std"]


def remap_table(variant):
    """LUT 0..255 -> classe 0..4. Valores fora do mapa viram 255 (invalido, detectado depois)."""
    t = np.full(256, 255, dtype=np.uint8)
    for src, dst in MASK_CLASS_MAPPING[variant].items():
        t[int(src)] = dst
    return t


class PatchDataset(Dataset):
    def __init__(self, csv_path, root, mean, std, variant):
        self.df = pd.read_csv(csv_path)
        self.root = root
        self.lut = remap_table(variant)
        self.tf = A.Compose([
            A.ToFloat(max_value=65535.0),
            A.Normalize(mean=mean, std=std, max_pixel_value=1.0),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        with rasterio.open(os.path.join(self.root, r["image"])) as s:
            img = np.transpose(s.read(SELECTED_BANDS), (1, 2, 0))   # (H,W,C) uint16, ja reordenado
        with rasterio.open(os.path.join(self.root, r["mask"])) as s:
            m = self.lut[s.read(1).astype(np.uint8)]                # subclasses 1..10 -> 0..4
        if (m == 255).any():
            raise ValueError("valor de rotulo fora do mask_class_mapping em %s" % r["mask"])
        out = self.tf(image=img, mask=m)
        return out["image"], out["mask"].long()


def build_model(arch, in_ch, n_classes):
    cls = {"Unet": smp.Unet, "DeepLabV3Plus": smp.DeepLabV3Plus, "Segformer": smp.Segformer}[arch]
    return cls(encoder_name=ENCODER[arch], encoder_weights=None,
               in_channels=in_ch, classes=n_classes, activation=None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds-root", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="globs das pastas de run (relativos a raiz)")
    ap.add_argument("--split", default="val", choices=["test", "val"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-patches", type=int, default=0, help="DEBUG: limita patches; grava *_DEBUG.json")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--force-test", action="store_true",
                    help="permite --split test em runs de screening (quebra o protocolo)")
    args = ap.parse_args()

    run_dirs = []
    for g in args.runs:
        run_dirs += sorted(glob.glob(g if os.path.isabs(g) else os.path.join(REPO, g)))
    run_dirs = [d for d in run_dirs if os.path.isdir(d)]
    print("%d run(s) | split=%s | device=%s" % (len(run_dirs), args.split, args.device))

    out_name = ({"test": "test_metrics", "val": "val_check_metrics"}[args.split]
                + ("_DEBUG" if args.max_patches else "") + ".json")
    done = 0
    for d in run_dirs:
        name = os.path.basename(d)
        m = RUN_RE.match(name)
        if not m:
            print("[nome nao reconhecido] %s" % name)
            continue
        info = m.groupdict()
        if info["phase"] == "s1" and args.split == "test" and not args.force_test:
            print("[recusado] %s e run de screening; pontue na validacao (--split val)" % name)
            continue

        out_path = os.path.join(d, out_name)
        if os.path.exists(out_path) and not args.overwrite:
            print("[pula] %s (%s ja existe)" % (name, out_name))
            continue
        ckpts = glob.glob(os.path.join(d, "checkpoints", "best-*.ckpt"))
        if not ckpts:
            print("[sem checkpoint] %s" % name)
            continue
        ckpt_path = sorted(ckpts)[-1]

        t0 = time.time()
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = {k[len("model."):]: v for k, v in ck["state_dict"].items() if k.startswith("model.")}
        model = build_model(info["arch"], len(SELECTED_BANDS), N_CLASSES)
        model.load_state_dict(sd, strict=True)
        model.to(args.device).eval()

        mean, std = band_stats(info["fold"])
        csv_path = os.path.join(REPO, "splits", "csv_derived", info["variant"],
                                "fold%s" % info["fold"], "%s.csv" % args.split)
        ds = PatchDataset(csv_path, args.ds_root, mean, std, info["variant"])
        if args.max_patches:
            ds.df = ds.df.iloc[: args.max_patches]
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

        kw = dict(task="multiclass", num_classes=N_CLASSES)      # SEM ignore_index
        macro = {"miou": tm.JaccardIndex(average="macro", **kw), "f1": tm.F1Score(average="macro", **kw),
                 "precision": tm.Precision(average="macro", **kw), "recall": tm.Recall(average="macro", **kw),
                 "accuracy": tm.Accuracy(average="macro", **kw)}
        for v in macro.values():
            v.to(args.device)
        cm = torch.zeros(N_CLASSES, N_CLASSES, dtype=torch.long, device=args.device)  # linha=alvo, coluna=predito
        with torch.no_grad():
            for x, y in dl:
                x, y = x.to(args.device).float(), y.to(args.device)
                pred = model(x).argmax(dim=1)
                for v in macro.values():
                    v.update(pred, y)
                cm += torch.bincount(y.reshape(-1) * N_CLASSES + pred.reshape(-1),
                                     minlength=N_CLASSES ** 2).view(N_CLASSES, N_CLASSES)
        tm_macro = {k: float(v.compute()) for k, v in macro.items()}

        tp = cm.diag().double()
        fn = cm.sum(1).double() - tp
        fp = cm.sum(0).double() - tp
        div = lambda a, b: torch.where(b > 0, a / b.clamp(min=1), torch.zeros_like(a))
        per = {"iou_class": div(tp, tp + fp + fn), "f1_class": div(2 * tp, 2 * tp + fp + fn),
               "precision_class": div(tp, tp + fp), "recall_class": div(tp, tp + fn)}
        present = cm.sum(1) > 0            # macro so sobre as classes presentes no alvo
        res = {}
        res["miou"] = float(per["iou_class"][present].mean())
        res["f1"] = float(per["f1_class"][present].mean())
        res["precision"] = float(per["precision_class"][present].mean())
        res["recall"] = float(per["recall_class"][present].mean())
        res["accuracy"] = res["recall"]     # acuracia balanceada = recall macro
        for mname, t in per.items():
            res[mname] = {str(i): float(t[i]) for i in range(N_CLASSES)}
        res["support_class"] = {str(i): int(cm[i].sum()) for i in range(N_CLASSES)}
        res["class_names"] = NOMES[info["variant"]]
        res["macro_vs_torchmetrics"] = {k: tm_macro[k] - res[k]
                                        for k in ("miou", "f1", "precision", "recall", "accuracy")}
        res.update(run=name, phase=info["phase"], loss=info["loss"], arch=info["arch"],
                   encoder=ENCODER[info["arch"]], lr=info["lr"], variant=info["variant"],
                   fold=int(info["fold"]), seed=int(info["seed"]), split=args.split,
                   n_classes=N_CLASSES, selected_bands=SELECTED_BANDS, n_patches=len(ds),
                   checkpoint=os.path.relpath(ckpt_path, REPO), ckpt_epoch=int(ck.get("epoch", -1)),
                   device=args.device, date=datetime.datetime.now().isoformat(timespec="seconds"))
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        done += 1
        print("[ok] %s: mIoU %.4f F1 %.4f (n=%d, %.0fs) -> %s"
              % (name, res["miou"], res["f1"], len(ds), time.time() - t0, out_name))
    print("pronto: %d run(s) avaliada(s)." % done)


if __name__ == "__main__":
    main()
