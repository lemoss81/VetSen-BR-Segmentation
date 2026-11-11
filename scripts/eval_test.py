# -*- coding: utf-8 -*-
"""
scripts/eval_test.py
  Evaluates the BEST checkpoint (lowest loss/val) of each run on the requested split
  (test by default) and writes <run>/test_metrics.json -- the input of scripts/aggregate_results.py.

  Metrics = same as training: macro, ignore_index=0 (NoData), torchmetrics:
  mIoU (JaccardIndex), F1, Precision, Recall, Accuracy + per-class IoU (1..5).
  Preprocessing = the val_dataset of configs/base_vetsenbr.yaml (ToFloat + Normalize),
  no augmentation. Inference in fp32, patch by patch (256x256), no sliding window.

  Usage (from the repo root, via pixi):
    pixi run python scripts/eval_test.py --ds-root <DS_ROOT> \
        --runs "experiments/r2_*" "experiments/r1_dice_focal_*"
    # reproduction check (should match the training val):
    pixi run python scripts/eval_test.py --ds-root <DS_ROOT> --split val --device cpu \
        --runs experiments/r1_dice_focal_Unet_fold0_seed42
"""
import argparse, datetime, glob, json, os, re, sys, time

import albumentations as A
import numpy as np
import pandas as pd
import rasterio
import segmentation_models_pytorch as smp
import torch
import torchmetrics as tm
from albumentations.pytorch import ToTensorV2
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_RE = re.compile(
    r"^(?P<phase>r[12])_(?:(?P<loss>.+?)_)?(?P<arch>Unet|DeepLabV3Plus)_"
    r"(?:(?P<variant>forest_nonforest|natural_cultivated)_)?fold(?P<fold>\d)_seed(?P<seed>\d+)$"
)


class PatchDataset(Dataset):
    def __init__(self, csv_path, root, mean, std):
        self.df = pd.read_csv(csv_path)
        self.root = root
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
            img = np.transpose(s.read(), (1, 2, 0))          # (H, W, C) uint16
        with rasterio.open(os.path.join(self.root, r["mask"])) as s:
            m = s.read(1).astype(np.uint8)
        out = self.tf(image=img, mask=m)
        return out["image"], out["mask"].long()


def build_model(arch, in_ch, n_classes):
    cls = {"Unet": smp.Unet, "DeepLabV3Plus": smp.DeepLabV3Plus}[arch]
    return cls(encoder_name="resnet101", encoder_weights=None, in_channels=in_ch,
               classes=n_classes, activation=None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds-root", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="globs of run folders (relative to the repo root)")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=0, help="torch.set_num_threads (0 = default)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-patches", type=int, default=0, help="DEBUG: limit patches; writes *_DEBUG.json")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    cfg = OmegaConf.load(os.path.join(REPO, "configs", "base_vetsenbr.yaml"))
    norm = [t for t in cfg.val_dataset.augmentation_list if str(t.get("_target_", "")).endswith("Normalize")][0]
    mean, std = list(norm.mean), list(norm.std)
    n_classes = int(cfg.hyperparameters.classes)
    in_ch = int(cfg.get("model", {}).get("in_channels", 10))

    run_dirs = []
    for g in args.runs:
        run_dirs += sorted(glob.glob(os.path.join(REPO, g)) if not os.path.isabs(g) else glob.glob(g))
    run_dirs = [d for d in run_dirs if os.path.isdir(d)]
    print(f"{len(run_dirs)} run(s) | split={args.split} | device={args.device}")

    out_name = {"test": "test_metrics", "val": "val_check_metrics"}[args.split] + ("_DEBUG" if args.max_patches else "") + ".json"
    done = 0
    for d in run_dirs:
        name = os.path.basename(d)
        m = RUN_RE.match(name)
        if not m:
            continue
        info = m.groupdict()
        info["variant"] = info["variant"] or "forest_nonforest"   # r1 only ran forest_nonforest
        out_path = os.path.join(d, out_name)
        if os.path.exists(out_path) and not args.overwrite:
            print(f"[skip] {name} ({out_name} already exists)"); continue
        ckpts = [c for c in glob.glob(os.path.join(d, "checkpoints", "best-*.ckpt"))]
        if not ckpts:
            print(f"[no checkpoint] {name}"); continue
        ckpt_path = sorted(ckpts)[-1]
        t0 = time.time()
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = {k[len("model."):]: v for k, v in ck["state_dict"].items() if k.startswith("model.")}
        model = build_model(info["arch"], in_ch, n_classes)
        model.load_state_dict(sd, strict=True)
        model.to(args.device).eval()

        csv_path = os.path.join(REPO, "splits", "csv", info["variant"], f"fold{info['fold']}", f"{args.split}.csv")
        ds = PatchDataset(csv_path, args.ds_root, mean, std)
        if args.max_patches:
            ds.df = ds.df.iloc[: args.max_patches]
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

        kw = dict(task="multiclass", num_classes=n_classes, ignore_index=0)
        macro = {"miou": tm.JaccardIndex(average="macro", **kw), "f1": tm.F1Score(average="macro", **kw),
                 "precision": tm.Precision(average="macro", **kw), "recall": tm.Recall(average="macro", **kw),
                 "accuracy": tm.Accuracy(average="macro", **kw)}
        for v in macro.values():
            v.to(args.device)
        cm = torch.zeros(n_classes, n_classes, dtype=torch.long, device=args.device)   # rows = target, cols = prediction
        with torch.no_grad():
            for x, y in dl:
                x, y = x.to(args.device).float(), y.to(args.device)
                pred = model(x).argmax(dim=1)
                for v in macro.values():
                    v.update(pred, y)
                valid = y != 0                                  # drop NoData, matching ignore_index=0
                idx = y[valid] * n_classes + pred[valid]
                cm += torch.bincount(idx, minlength=n_classes ** 2).view(n_classes, n_classes)
        tm_macro = {k: float(v.compute()) for k, v in macro.items()}   # torchmetrics macro (reference)
        # per-class metrics (1..5) from the globally accumulated confusion matrix
        tp = cm.diag().double()
        fn = cm.sum(1).double() - tp
        fp = cm.sum(0).double() - tp
        div = lambda a, b: torch.where(b > 0, a / b.clamp(min=1), torch.zeros_like(a))
        per = {"iou_class": div(tp, tp + fp + fn), "f1_class": div(2 * tp, 2 * tp + fp + fn),
               "precision_class": div(tp, tp + fp), "recall_class": div(tp, tp + fn)}
        # macro = mean over the REAL classes present in the target (1..5), excluding NoData (class 0).
        # Fixes the torchmetrics artifact that counts class 0 in the denominator whenever the model
        # predicts any NoData pixel (dividing by 6 instead of 5) for F1/Precision/Recall.
        present = cm.sum(1)[1:] > 0
        res = {}
        res["miou"]      = float(per["iou_class"][1:][present].mean())
        res["f1"]        = float(per["f1_class"][1:][present].mean())
        res["precision"] = float(per["precision_class"][1:][present].mean())
        res["recall"]    = float(per["recall_class"][1:][present].mean())
        res["accuracy"]  = res["recall"]   # macro (balanced) accuracy = macro recall in the multiclass case
        for mname, t in per.items():
            res[mname] = {str(i): float(t[i]) for i in range(1, n_classes)}
        res["support_class"] = {str(i): int(cm[i].sum()) for i in range(1, n_classes)}   # valid pixels per class
        # difference vs the torchmetrics macro (informative; != 0 for runs that predict NoData)
        res["macro_vs_torchmetrics"] = {k: tm_macro[k] - res[k] for k in ("miou", "f1", "precision", "recall", "accuracy")}
        res.update(run=name, phase=info["phase"], loss=info["loss"], arch=info["arch"], variant=info["variant"],
                   fold=int(info["fold"]), seed=int(info["seed"]), split=args.split, n_patches=len(ds),
                   checkpoint=os.path.relpath(ckpt_path, REPO), ckpt_epoch=int(ck.get("epoch", -1)),
                   device=args.device, date=datetime.datetime.now().isoformat(timespec="seconds"))
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        done += 1
        print(f"[ok] {name}: mIoU {res['miou']:.4f} F1 {res['f1']:.4f} (n={len(ds)}, {time.time()-t0:.0f}s, "
              f"dif F1 vs torchmetrics: {res['macro_vs_torchmetrics']['f1']:+.1e}) -> {out_name}")
    print(f"done: {done} run(s) evaluated.")


if __name__ == "__main__":
    main()
