# -*- coding: utf-8 -*-
r"""
build_kaggle_upload.py — assembles the VetSen-BR upload package for Kaggle.

The on-disk dataset (D:\v2\dataset) already uses the "one image, N labels" layout
(shared images/ + labels/<variant>/). This script only STAGES a clean upload package,
validating counts and name pairing, and gathers the READMEs, the splits and
dataset-metadata.json.

By default it uses HARD LINKS (files are not duplicated; this only works if OUT_DIR
is on the SAME volume as SRC_DS — both on D:). If linking fails (different volume),
it falls back to copying automatically. Idempotent: skips files that already exist
at the destination.

    python build_kaggle_upload.py
    python build_kaggle_upload.py --copy            # force copy instead of hard link
    python build_kaggle_upload.py --out E:\kaggle   # other destination (becomes a copy)

It uploads nothing. At the end it prints the Kaggle CLI command for you to run.
"""
import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

# ------------------------------------------------------------------ CONFIG
SRC_DS     = Path(r"D:\v2\dataset")                        # images/ + labels/<variant>/
SPLITS_SRC = Path(r"C:\git\VetSen-BR-Segmentation\splits") # catalog/folds + csv/<variant>/
OUT_DIR    = Path(r"D:\v2\kaggle_upload")                  # staged package (same volume as SRC -> hard link)
VARIANTS   = ["forest_nonforest", "natural_cultivated", "subclasses"]  # add "natural_cultivated_rock" if needed
N_EXPECTED = 9320
SELF_DIR   = Path(__file__).resolve().parent              # kaggle/ in the repo (READMEs and metadata)
# README_<variant>.md -> labels/<variant>/README.md ; README.md -> root
# -------------------------------------------------------------------------


def _link_or_copy(src: Path, dst: Path, mode: str):
    """Hard link (mode='link') with copy fallback; 'copy' always copies. Skips if it already exists."""
    if dst.exists():
        return "skip"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "link":
        try:
            os.link(src, dst)
            return "link"
        except OSError:
            pass  # different volume / FS without hard links -> copy
    shutil.copy2(src, dst)
    return "copy"


def _list_tifs(d: Path):
    return sorted(p.name for p in d.glob("*.tif")) if d.is_dir() else []


def validar():
    erros = []
    if not SRC_DS.is_dir():
        erros.append(f"SRC_DS does not exist: {SRC_DS}")
        return erros, {}
    imgs = _list_tifs(SRC_DS / "images")
    if len(imgs) != N_EXPECTED:
        erros.append(f"images/: {len(imgs)} tifs (expected {N_EXPECTED})")
    img_set = set(imgs)
    por_var = {"images": imgs}
    for v in VARIANTS:
        labs = _list_tifs(SRC_DS / "labels" / v)
        por_var[v] = labs
        if len(labs) != N_EXPECTED:
            erros.append(f"labels/{v}/: {len(labs)} tifs (expected {N_EXPECTED})")
        faltam = img_set - set(labs)
        sobram = set(labs) - img_set
        if faltam:
            erros.append(f"labels/{v}/: {len(faltam)} names without an image (e.g.: {sorted(faltam)[:3]})")
        if sobram:
            erros.append(f"labels/{v}/: {len(sobram)} names without a matching label (e.g.: {sorted(sobram)[:3]})")
    # The split is PER CHART and identical for all variants (same images). The per-variant
    # csv only changes the mask path; if missing, folds_por_patch.csv already defines the fold.
    warns = []
    for v in VARIANTS:
        csvdir = SPLITS_SRC / "csv" / v
        if not csvdir.is_dir():
            warns.append(f"no csv/{v}/ (per-chart fold is in folds_por_patch.csv; mask = labels/{v}/<name>)")
    return erros, warns, por_var


def copiar_dados(mode: str, por_var: dict):
    stats = {"link": 0, "copy": 0, "skip": 0}

    def _bump(r):
        stats[r] += 1

    # images (once)
    print("  -> images/ ...", flush=True)
    for name in por_var["images"]:
        _bump(_link_or_copy(SRC_DS / "images" / name, OUT_DIR / "images" / name, mode))

    # labels per variant
    for v in VARIANTS:
        print(f"  -> labels/{v}/ ...", flush=True)
        for name in por_var[v]:
            _bump(_link_or_copy(SRC_DS / "labels" / v / name, OUT_DIR / "labels" / v / name, mode))
    return stats


def copiar_splits():
    dst = OUT_DIR / "splits"
    dst.mkdir(parents=True, exist_ok=True)
    for f in ["catalog.csv", "folds_por_carta.csv", "folds_por_patch.csv"]:
        src = SPLITS_SRC / f
        if src.exists():
            shutil.copy2(src, dst / f)
    # csv/<variant>/fold*/...  (copy only variants that have a per-variant split)
    for v in VARIANTS:
        s = SPLITS_SRC / "csv" / v
        if s.is_dir():
            shutil.copytree(s, dst / "csv" / v, dirs_exist_ok=True)


def copiar_readmes_e_meta():
    # root README
    root_rm = SELF_DIR / "README.md"
    if root_rm.exists():
        shutil.copy2(root_rm, OUT_DIR / "README.md")
    # per-variant README -> labels/<variant>/README.md
    for v in VARIANTS:
        rm = SELF_DIR / f"README_{v}.md"
        if rm.exists():
            (OUT_DIR / "labels" / v).mkdir(parents=True, exist_ok=True)
            shutil.copy2(rm, OUT_DIR / "labels" / v / "README.md")
    # dataset-metadata.json (Kaggle CLI) at the package root
    meta = SELF_DIR / "dataset-metadata.json"
    if meta.exists():
        shutil.copy2(meta, OUT_DIR / "dataset-metadata.json")
    else:
        (OUT_DIR / "dataset-metadata.json").write_text(
            json.dumps({"title": "VetSen-BR", "id": "datasetbr1/vetsen-br",
                        "licenses": [{"name": "MIT"}]}, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--copy", action="store_true", help="force copy (instead of hard link)")
    ap.add_argument("--out", default=None, help="output directory (default D:\\v2\\kaggle_upload)")
    args = ap.parse_args()

    global OUT_DIR
    if args.out:
        OUT_DIR = Path(args.out)
    mode = "copy" if args.copy else "link"

    print(f"[VALIDATE] SRC={SRC_DS}  variants={VARIANTS}")
    erros, warns, por_var = validar()
    if erros:
        print("\n[FAILED] inconsistencies found:")
        for e in erros:
            print("   -", e)
        print("\nFix the source before assembling the package. Nothing was written.")
        sys.exit(1)
    for w in warns:
        print("   [warning]", w)
    print(f"[OK] images and {len(VARIANTS)} variants with {N_EXPECTED} tifs, names paired.")

    print(f"[ASSEMBLE] OUT={OUT_DIR}  mode={'HARD LINK (copy fallback)' if mode=='link' else 'COPY'}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stats = copiar_dados(mode, por_var)
    copiar_splits()
    copiar_readmes_e_meta()

    n_img = len(por_var["images"])
    print("\n[DONE]")
    print(f"   files: link={stats['link']}  copy={stats['copy']}  skip={stats['skip']}")
    print(f"   package at: {OUT_DIR}")
    print(f"   structure: images/ ({n_img})  labels/{{{','.join(VARIANTS)}}}/  splits/  README.md")
    print("\n[UPLOAD] (new version, keeps the URL /datasetbr1/vetsen-br):")
    print(f'   kaggle datasets version -p "{OUT_DIR}" -m "v16: 9320 patches, new class naming, dedup layout" --dir-mode zip')


if __name__ == "__main__":
    main()
