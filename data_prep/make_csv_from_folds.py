# -*- coding: utf-8 -*-
r"""
data_prep/make_csv_from_folds.py
  Builds the (image,mask) CSVs consumed by pytorch-segmentation-models-trainer from the
  chart-grouped folds, using a 70/10/20 scheme GROUPED BY CHART (no leakage):
      TEST  = fold f  (~20%)
      VAL   = ~12.5% of the TRAIN charts, stratified by biome (-> ~10% of the total)
      TRAIN = the rest (~70%)
  For each VARIANT (forest_nonforest, natural_cultivated) and each fold (0..4).
  Reads splits/folds_por_patch.csv (sNNN) and splits/folds_por_carta.csv. Fully deterministic.

  Output: splits/csv/<variant>/fold<f>/{train,val,test}.csv
"""
import csv
import argparse
from pathlib import Path
from collections import defaultdict

K = 5
BASE = Path(__file__).resolve().parent.parent / "splits"
VAL_FRAC_OF_TRAIN = 0.125   # 0.10 / 0.80  -> val ~10% of the total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["forest_nonforest", "natural_cultivated"])
    ap.add_argument("--out", default=str(BASE / "csv"))
    args = ap.parse_args()

    # patch -> (chart, fold) ; chart -> patches ; chart -> fold
    patches_por_carta = defaultdict(list)
    carta_fold = {}
    with open(BASE / "folds_por_patch.csv", encoding="utf-8") as f:
        for d in csv.DictReader(f):
            c = int(d["carta"]); patches_por_carta[c].append(d["patch"]); carta_fold[c] = int(d["fold"])
    # chart -> biome
    carta_bioma = {}
    with open(BASE / "folds_por_carta.csv", encoding="utf-8") as f:
        for d in csv.DictReader(f):
            carta_bioma[int(d["carta"])] = d["bioma"]
    npatch = {c: len(v) for c, v in patches_por_carta.items()}
    total = sum(npatch.values())
    print(f"patches: {total} | charts: {len(npatch)}")

    def carve_val(train_cartas):
        """Pick validation charts (~12.5% of the train patches), stratified by biome, deterministic."""
        val = set()
        by_b = defaultdict(list)
        for c in train_cartas:
            by_b[carta_bioma[c]].append(c)
        for b, cs in by_b.items():
            cs = sorted(cs, key=lambda c: (npatch[c], c))   # smallest first, deterministic
            target = VAL_FRAC_OF_TRAIN * sum(npatch[c] for c in cs)
            acc = 0
            for c in cs:
                if acc >= target:
                    break
                val.add(c); acc += npatch[c]
        return val

    def write_csv(path, cartas, variant):
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = sorted(p for c in cartas for p in patches_por_carta[c])
        with open(path, "w", newline="", encoding="utf-8") as fo:
            w = csv.writer(fo); w.writerow(["image", "mask"])
            for p in rows:
                w.writerow([f"images/{p}", f"labels/{variant}/{p}"])
        return len(rows)

    for variant in args.variants:
        print(f"\n== {variant} ==")
        for f in range(K):
            test_c = [c for c in npatch if carta_fold[c] == f]
            train_all = [c for c in npatch if carta_fold[c] != f]
            val_c = carve_val(train_all)
            train_c = [c for c in train_all if c not in val_c]
            d = Path(args.out) / variant / f"fold{f}"
            ntr = write_csv(d / "train.csv", train_c, variant)
            nva = write_csv(d / "val.csv", sorted(val_c), variant)
            nte = write_csv(d / "test.csv", test_c, variant)
            print(f"  fold{f}: train={ntr:5} ({100*ntr/total:4.1f}%) val={nva:5} ({100*nva/total:4.1f}%) "
                  f"test={nte:5} ({100*nte/total:4.1f}%)")
    print(f"\nCSVs in {args.out}")


if __name__ == "__main__":
    main()
