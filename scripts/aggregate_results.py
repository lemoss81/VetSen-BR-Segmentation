# -*- coding: utf-8 -*-
r"""
scripts/aggregate_results.py
  Agrega as métricas das runs (média±desvio por configuração) e, com --compare-variants,
  aplica o teste pareado entre as variantes (Wilcoxon se >=6 pares; senão t pareado).

  ⚠️ Ajuste METRIC_FILE_GLOB / get_metric() ao formato real de saída do PSMT no 1º run
     (ex.: experiments/<run>/.../summary_report*.json com chaves de IoU/F1 no teste).

    python scripts/aggregate_results.py --glob "experiments/r2_*" --compare-variants
"""
import os, re, csv, json, glob, argparse, statistics as st
from collections import defaultdict

METRIC_FILE_GLOB = ["**/summary_report*.json", "**/aggregated_metrics.json", "**/test*metrics*.json"]
RUN_RE = re.compile(r"(?P<arch>Unet|DeepLabV3Plus)_(?P<variant>forest_nonforest|natural_cultivated)_fold(?P<fold>\d)_seed(?P<seed>\d+)")


def find_metric_file(run_dir):
    for pat in METRIC_FILE_GLOB:
        hits = glob.glob(os.path.join(run_dir, pat), recursive=True)
        if hits:
            return sorted(hits)[0]
    return None


def get_metric(d, want):
    """Procura recursivamente a 1ª chave cujo nome contém `want` (ex.: 'iou','f1') e valor numérico."""
    want = want.lower()
    out = []
    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if want in str(k).lower() and isinstance(v, (int, float)):
                    out.append(float(v))
                walk(v)
        elif isinstance(o, list):
            for v in o: walk(v)
    walk(d)
    return out[0] if out else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="glob dos dirs de run, ex.: 'experiments/r2_*'")
    ap.add_argument("--compare-variants", action="store_true")
    args = ap.parse_args()

    rows = []
    for run_dir in sorted(glob.glob(args.glob)):
        m = RUN_RE.search(os.path.basename(run_dir))
        if not m:
            continue
        mf = find_metric_file(run_dir)
        if not mf:
            print(f"  [sem métricas] {run_dir}"); continue
        d = json.load(open(mf, encoding="utf-8"))
        rows.append(dict(**m.groupdict(), miou=get_metric(d, "iou"), f1=get_metric(d, "f1")))

    if not rows:
        print("nenhuma run encontrada / métricas não lidas — confira METRIC_FILE_GLOB/get_metric()."); return

    # média±desvio por (arch, variant)
    print("\n== média±desvio por (arch, variant) ==")
    cells = defaultdict(list)
    for r in rows:
        cells[(r["arch"], r["variant"])].append(r)
    for (arch, v), rs in sorted(cells.items()):
        mi = [x["miou"] for x in rs if x["miou"] is not None]
        f1 = [x["f1"] for x in rs if x["f1"] is not None]
        print(f"  {arch:14} {v:18} n={len(rs):2} | mIoU {st.mean(mi):.4f}±{st.pstdev(mi):.4f} | F1 {st.mean(f1):.4f}±{st.pstdev(f1):.4f}")

    if args.compare_variants:
        import scipy.stats as ss
        print("\n== comparação de variantes (pareado por arch,fold,seed) ==")
        for arch in sorted({r["arch"] for r in rows}):
            idx = lambda r: (r["fold"], r["seed"])
            a = {idx(r): r["miou"] for r in rows if r["arch"] == arch and r["variant"] == "forest_nonforest"}
            b = {idx(r): r["miou"] for r in rows if r["arch"] == arch and r["variant"] == "natural_cultivated"}
            keys = sorted(set(a) & set(b))
            da = [a[k] for k in keys]; db = [b[k] for k in keys]
            if len(keys) < 2:
                print(f"  {arch}: pares insuficientes"); continue
            if len(keys) >= 6:
                stat, p = ss.wilcoxon(da, db); test = "Wilcoxon"
            else:
                stat, p = ss.ttest_rel(da, db); test = "t pareado"
            print(f"  {arch}: n={len(keys)} | FNF {st.mean(da):.4f} vs NC {st.mean(db):.4f} | {test} p={p:.4f}")


if __name__ == "__main__":
    main()
