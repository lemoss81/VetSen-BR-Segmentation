# -*- coding: utf-8 -*-
r"""
data_prep/build_catalog.py  (roda UMA vez, na máquina que tem o cartas_indice)
  Gera os artefatos de rastreabilidade do benchmark VetSen-BR:
    - filename_map.csv : nome_antigo -> nome_novo  (sNNN)
    - catalog.csv      : por folha -> sNNN, INOM, MI, nome, regiao, bioma, data_imageamento, n_patches, fold
    - folds_por_patch.csv  (nomes sNNN): patch, sheet_id, carta, fold
    - folds_por_carta.csv  (com sheet_id)
  Convenção de nome:  carta{NN}_img{K}_{I}_{J}.tif  ->  VetSenBR_s{NNN}_a{K}_r{I:04d}_c{J:04d}.tif
"""
import os, re, csv, sys
from pathlib import Path
_e = Path(sys.executable).parent
for k, s in (("PROJ_LIB", "Library/share/proj"), ("GDAL_DATA", "Library/share/gdal")):
    if (_e / s).exists():
        os.environ[k] = str(_e / s)
try:
    os.add_dll_directory(str(_e / "Library/bin"))
except Exception:
    pass
os.environ["PATH"] = str(_e / "Library/bin") + os.pathsep + os.environ.get("PATH", "")
import warnings; warnings.filterwarnings("ignore")
import geopandas as gpd

DSREPO = Path(r"C:\git\VetSenBR_dataset")
SPLITS_IN = DSREPO / "splits"
OUT = Path(r"C:\git\VetSen-BR-Segmentation\splits")
sys.path.insert(0, str(DSREPO)); import config

PAT = re.compile(r"carta(\d+)_img(\d+)_(\d+)_(\d+)\.tif$", re.I)


def novo_nome(old):
    m = PAT.match(old)
    if not m:
        return None
    nn, k, i, j = (int(x) for x in m.groups())
    return f"VetSenBR_s{nn:03d}_a{k}_r{i:04d}_c{j:04d}.tif"


def cidade(nome):
    parts = str(nome).split("-")
    return parts[1].strip() if len(parts) > 1 else str(nome)


OUT.mkdir(parents=True, exist_ok=True)

# cartas_indice -> atributos por carta
g = gpd.read_file(str(config.SHP_SELECAO_ENTRADA))
attr = {}
for _, r in g.iterrows():
    try:
        num = int(str(r["Nome"]).split("_")[0])
    except Exception:
        continue
    attr[num] = dict(inom=r["inom"], mi=r["mi"], regiao=r["Regiao"],
                     nome=cidade(r["Nome"]), imageamento=r["Imageament"])

# folds_por_carta (bioma, fold, n_patches)
carta_info = {}
with open(SPLITS_IN / "folds_por_carta.csv", encoding="utf-8") as f:
    for d in csv.DictReader(f):
        carta_info[int(d["carta"])] = d

# filename_map + folds_por_patch(sNNN)
fmap = []
fpp = []
with open(SPLITS_IN / "folds_por_patch.csv", encoding="utf-8") as f:
    for d in csv.DictReader(f):
        old = d["patch"]; new = novo_nome(old); carta = int(d["carta"])
        if new is None:
            print("  [aviso] nome fora do padrão:", old); continue
        fmap.append((old, new))
        fpp.append((new, f"s{carta:03d}", carta, d["fold"]))

with open(OUT / "filename_map.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["old_name", "new_name"]); w.writerows(fmap)
with open(OUT / "folds_por_patch.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["patch", "sheet_id", "carta", "fold"]); w.writerows(fpp)

# catalog por folha
with open(OUT / "catalog.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["sheet_id", "carta", "inom", "mi", "nome_folha", "regiao", "bioma", "data_imageamento", "n_patches", "fold"])
    for num in sorted(carta_info):
        a = attr.get(num, {}); ci = carta_info[num]
        w.writerow([f"s{num:03d}", num, a.get("inom", ""), a.get("mi", ""), a.get("nome", ""),
                    a.get("regiao", ""), ci.get("bioma", ""), a.get("imageamento", ""),
                    ci.get("n_patches", ""), ci.get("fold", "")])

# copia folds_por_carta com sheet_id
with open(SPLITS_IN / "folds_por_carta.csv", encoding="utf-8") as f, \
     open(OUT / "folds_por_carta.csv", "w", newline="", encoding="utf-8") as fo:
    rd = csv.DictReader(f); w = csv.writer(fo)
    w.writerow(["sheet_id"] + rd.fieldnames)
    for d in rd:
        w.writerow([f"s{int(d['carta']):03d}"] + [d[k] for k in rd.fieldnames])

print(f"patches mapeados: {len(fmap)} | folhas no catálogo: {len(carta_info)}")
print("exemplo:", fmap[0][0], "->", fmap[0][1])
print("salvo em", OUT)
