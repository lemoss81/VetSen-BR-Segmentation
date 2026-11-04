# -*- coding: utf-8 -*-
r"""
scripts/aggregate_results.py
  Agrega as metricas por run (media +- desvio) e compara variantes / arquiteturas com teste pareado
  (Wilcoxon se >= 6 pares, senao t pareado). Le <run>/test_metrics.json, gerado por scripts/eval_test.py.
  Metricas globais (macro) e POR CLASSE (IoU, F1, precisao, recall das classes 1..5).

  A celula forest_nonforest/seed 42 da Run 2 vem das runs da Run 1 (mesma loss e config):
  inclua as duas familias de pastas. Desvio = amostral (ddof=1).
  Nomes das classes (VetSen-BR-dataset, classificador_vetores_EDGV.py): 1 Water, 2 Built-up, 5 Barren Land
  em ambas as variantes; 3/4 = Forest/Non-forest (forest_nonforest) ou Natural/Cultivated (natural_cultivated).
  Entre variantes, a comparacao por classe so e feita nas classes comuns (1, 2, 5); 3 e 4 mudam de significado.

  Run 2 (principal):
    python scripts/aggregate_results.py --glob "experiments/r2_*" --glob "experiments/r1_dice_focal_*" \
        --loss dice_focal --compare-variants --compare-archs --out experiments/resultado_final

  Run 1 (ranking das losses; use --split-file val_check_metrics.json se avaliou na validacao):
    python scripts/aggregate_results.py --glob "experiments/r1_*" --by-loss --split-file val_check_metrics.json

"""
import argparse, csv, glob, json, os, statistics as st
from collections import defaultdict

CLASSES = ["1", "2", "3", "4", "5"]
NAMES = {"forest_nonforest": {"1": "Water", "2": "Built-up", "3": "Forest", "4": "Non-forest", "5": "Barren"},
         "natural_cultivated": {"1": "Water", "2": "Built-up", "3": "Natural", "4": "Cultivated", "5": "Barren"}}
SHARED = ["1", "2", "5"]
PER_CLASS = [("iou_class", "IoU"), ("f1_class", "F1"), ("precision_class", "P"), ("recall_class", "R")]


def sd(x):
    return st.stdev(x) if len(x) > 1 else float("nan")


def load_rows(globs, default_loss, split_file):
    rows = []
    for g in globs:
        for d in sorted(glob.glob(g)):
            f = os.path.join(d, split_file)
            if not os.path.isfile(f):
                continue
            r = json.load(open(f, encoding="utf-8"))
            if not r.get("loss"):
                r["loss"] = default_loss          # runs r2_* nao trazem a loss no nome
            rows.append(r)
    return rows


def fmt(x):
    return f"{st.mean(x):.4f}+-{sd(x):.4f}"


def paired(a, b, label_a, label_b, ss):
    keys = sorted(set(a) & set(b))
    da, db = [a[k] for k in keys], [b[k] for k in keys]
    if len(keys) < 2:
        return f"n pares insuficiente ({len(keys)})"
    if all(x == y for x, y in zip(da, db)):
        return f"n={len(keys)} | valores identicos"
    if len(keys) >= 6:
        _, p = ss.wilcoxon(da, db); t = "Wilcoxon"
    else:
        _, p = ss.ttest_rel(da, db); t = "t pareado"
    return f"n={len(keys)} | {label_a} {st.mean(da):.4f} vs {label_b} {st.mean(db):.4f} | diff {st.mean(da)-st.mean(db):+.4f} | {t} p={p:.4g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", action="append", required=True, help="glob de pastas de run (repetivel)")
    ap.add_argument("--loss", default="dice_focal", help="loss das runs r2_* e filtro (ignorado com --by-loss)")
    ap.add_argument("--split-file", default="test_metrics.json")
    ap.add_argument("--by-loss", action="store_true")
    ap.add_argument("--compare-variants", action="store_true")
    ap.add_argument("--compare-archs", action="store_true")
    ap.add_argument("--out", default=None, help="prefixo de saida: <out>_por_run.csv e <out>_resumo.csv")
    args = ap.parse_args()

    def nm(variant, c):
        return f"{c}:{NAMES.get(variant, {}).get(c, c)}"

    rows = load_rows(args.glob, args.loss, args.split_file)
    if not args.by_loss:
        rows = [r for r in rows if r["loss"] == args.loss]
    if not rows:
        print(f"nenhuma run com {args.split_file} (rode scripts/eval_test.py antes)."); return
    print(f"{len(rows)} run(s) lidas ({args.split_file}).")

    key = (lambda r: (r["loss"], r["arch"], r["variant"])) if args.by_loss else (lambda r: (r["arch"], r["variant"]))
    cells = defaultdict(list)
    for r in rows:
        cells[key(r)].append(r)

    def cls_vals(rs, field, c):
        return [x[field][c] for x in rs if c in x.get(field, {})]

    print("\n== media +- desvio por celula ==")
    summary = []
    for k, rs in sorted(cells.items()):
        m = {n: [x[n] for x in rs] for n in ("miou", "f1", "precision", "recall", "accuracy")}
        tag = " ".join(f"{x:<14}" for x in k)
        variant = k[-1]
        print(f"{tag} n={len(rs):2} | mIoU {fmt(m['miou'])} | F1 {fmt(m['f1'])} | P {fmt(m['precision'])} | R {fmt(m['recall'])}")
        row = dict(zip(("loss", "arch", "variant") if args.by_loss else ("arch", "variant"), k), n=len(rs),
                   **{f"{n}_mean": st.mean(v) for n, v in m.items()}, **{f"{n}_std": sd(v) for n, v in m.items()})
        for field, lab in PER_CLASS:
            parts = []
            for c in CLASSES:
                v = cls_vals(rs, field, c)
                if v:
                    parts.append(f"{nm(variant, c)}={st.mean(v):.3f}+-{sd(v):.3f}")
                    row[f"{lab}_c{c}_mean"] = st.mean(v); row[f"{lab}_c{c}_std"] = sd(v)
            if parts:
                print(f"   {lab:>3} por classe: " + "  ".join(parts))
        summary.append(row)

    if args.compare_variants or args.compare_archs:
        import scipy.stats as ss
    if args.compare_variants:
        print("\n== variantes (pareado por arquitetura, fold, seed; mIoU e classes comuns 1, 2, 5) ==")
        for arch in sorted({r["arch"] for r in rows}):
            def pickv(v, f, arch=arch):
                return {(r["fold"], r["seed"]): f(r) for r in rows if r["arch"] == arch and r["variant"] == v}
            print(f"  {arch}:")
            print("    mIoU          : " + paired(pickv("forest_nonforest", lambda r: r["miou"]),
                                                  pickv("natural_cultivated", lambda r: r["miou"]), "FNF", "NC", ss))
            for field, lab in PER_CLASS[:2]:                      # IoU e F1 nas classes comuns
                for c in SHARED:
                    fa = lambda r, field=field, c=c: r.get(field, {}).get(c)
                    a = {k: x for k, x in pickv("forest_nonforest", fa).items() if x is not None}
                    b = {k: x for k, x in pickv("natural_cultivated", fa).items() if x is not None}
                    if a and b:
                        print(f"    {lab:>3} {nm('forest_nonforest', c):<10}: " + paired(a, b, "FNF", "NC", ss))
    if args.compare_archs:
        print("\n== arquiteturas (pareado por variante, fold, seed) ==")
        for v in sorted({r["variant"] for r in rows}):
            def pick(arch, f):
                return {(r["fold"], r["seed"]): f(r) for r in rows if r["variant"] == v and r["arch"] == arch}
            print(f"  {v}:")
            print("    mIoU        : " + paired(pick("Unet", lambda r: r["miou"]), pick("DeepLabV3Plus", lambda r: r["miou"]), "U-Net", "DeepLabV3+", ss))
            for field, lab in PER_CLASS[:2]:                      # IoU e F1 por classe
                for c in CLASSES:
                    fa = lambda r, field=field, c=c: r.get(field, {}).get(c)
                    a = {k: x for k, x in pick("Unet", fa).items() if x is not None}
                    b = {k: x for k, x in pick("DeepLabV3Plus", fa).items() if x is not None}
                    if a and b:
                        print(f"    {lab:>3} {nm(v, c):<14}: " + paired(a, b, "U-Net", "DeepLabV3+", ss))

    if args.out:
        cols = ["run", "loss", "arch", "variant", "fold", "seed", "miou", "f1", "precision", "recall", "accuracy", "ckpt_epoch"]
        pc_cols = [(field, lab, c) for field, lab in PER_CLASS for c in CLASSES]
        with open(args.out + "_por_run.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(cols + [f"{lab}_c{c}" for _, lab, c in pc_cols] + [f"support_c{c}" for c in CLASSES])
            for r in sorted(rows, key=lambda r: (r["arch"], r["variant"], r["fold"], r["seed"])):
                w.writerow([r.get(c) for c in cols] + [r.get(field, {}).get(c) for field, _, c in pc_cols]
                           + [r.get("support_class", {}).get(c) for c in CLASSES])
        with open(args.out + "_resumo.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for s in summary for k in s}))
            w.writeheader(); w.writerows(summary)
        print(f"\ngravado: {args.out}_por_run.csv e {args.out}_resumo.csv")


if __name__ == "__main__":
    main()
