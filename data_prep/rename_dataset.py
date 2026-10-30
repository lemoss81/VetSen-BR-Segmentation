# -*- coding: utf-8 -*-
r"""
data_prep/rename_dataset.py   (roda na máquina/SSD que tem o DATASET)
  Renomeia os arquivos do dataset para o padrão sNNN usando splits/filename_map.csv,
  em images/ e em TODAS as pastas de rótulo. DRY-RUN por padrão; use --apply.

    python data_prep/rename_dataset.py --root <SSD>/dataset
    python data_prep/rename_dataset.py --root <SSD>/dataset --apply
"""
import os, csv, argparse
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SUBDIRS = ["images", "images_tci",
           "labels/forest_nonforest", "labels/natural_cultivated",
           "labels/natural_cultivated_rock", "labels/subclasses"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dir do dataset (com images/ e labels/)")
    ap.add_argument("--map", default=str(REPO / "splits" / "filename_map.csv"))
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)

    mp = {}
    with open(args.map, encoding="utf-8") as f:
        for d in csv.DictReader(f):
            mp[d["old_name"]] = d["new_name"]
    print(f"[rename] mapa: {len(mp)} nomes | {'APLICAR' if args.apply else 'DRY-RUN'}")

    for sub in SUBDIRS:
        d = root / sub
        if not d.is_dir():
            print(f"  (pulado, não existe) {sub}"); continue
        feitos = faltando = 0
        for old, new in mp.items():
            po, pn = d / old, d / new
            if po.exists():
                if args.apply:
                    po.rename(pn)
                feitos += 1
                for ext in (".aux.xml",):  # sidecars
                    s = d / (old + ext)
                    if s.exists() and args.apply:
                        s.rename(d / (new + ext))
            elif not pn.exists():
                faltando += 1
        print(f"  {sub}: {feitos} {'renomeados' if args.apply else 'a renomear'}" +
              (f" | {faltando} ausentes" if faltando else ""))
    if not args.apply:
        print("\nDRY-RUN — nada mudou. Rode com --apply.")


if __name__ == "__main__":
    main()
