# -*- coding: utf-8 -*-
r"""
data_prep/make_derived_labels.py
  Gera os CSVs do treino com a coluna 'mask' apontando para labels/subclasses/ e
  documenta o mask_class_mapping de cada variante.

  POR QUE: as mascaras publicadas de forest_nonforest tem 8.275 pixels (54 patches)
  em que Silviculture ficou como Non-forest, por causa da prioridade de sobreposicao
  na rasterizacao (o esquema agregado rasterizava separado do subclasses). Derivar a
  variante do subclasses por LUT, na leitura, reproduz exatamente o que o pipeline
  corrigido produzira, e torna o treino imune a essa correcao do dataset.

  VERIFICADO sobre os 9.320 patches da v1.0.0:
    LUT(subclasses) vs natural_cultivated publicado -> 0 pixels diferentes
    LUT(subclasses) vs forest_nonforest  publicado -> 8.275 pixels, em 54 patches

  Os patches e os splits sao EXATAMENTE os de splits/csv/ (so muda a coluna 'mask').

  Uso:  python data_prep/make_derived_labels.py
"""
import csv
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "splits" / "csv"
DST = REPO / "splits" / "csv_derived"

# subclasses (1..10) -> classe final JA REMAPEADA para 0..4 (hyperparameters.classes = 5).
# Atencao: a classe 0 e Water, NAO e NoData. O dataset nao tem nenhum pixel NoData.
MASK_CLASS_MAPPING = {
    "forest_nonforest":   {1: 0, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 2, 8: 3, 9: 4, 10: 4},
    "natural_cultivated": {1: 0, 2: 1, 3: 2, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3, 9: 4, 10: 4},
}
NOMES = {
    "forest_nonforest":   ["Water", "Built-up", "Forest", "Non-forest", "Barren Land"],
    "natural_cultivated": ["Water", "Built-up", "Natural", "Cultivated", "Barren Land"],
}
# origem (subclasses) de cada classe final, para o README
ORIGEM = {1: "Water", 2: "Built-up", 3: "Natural forest", 4: "Savanna", 5: "Grassland",
          6: "Wetland", 7: "Silviculture", 8: "Cropland", 9: "Bare soil", 10: "Rock"}


def main():
    total = 0
    for variant in MASK_CLASS_MAPPING:
        for fold in range(5):
            for split in ("train", "val", "test"):
                src = SRC / variant / f"fold{fold}" / f"{split}.csv"
                dst = DST / variant / f"fold{fold}" / f"{split}.csv"
                dst.parent.mkdir(parents=True, exist_ok=True)
                with open(src, encoding="utf-8", newline="") as f:
                    rows = list(csv.DictReader(f))
                with open(dst, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["image", "mask"])
                    for r in rows:
                        name = os.path.basename(r["mask"])
                        w.writerow([r["image"], f"labels/subclasses/{name}"])
                total += 1
                print(f"  {dst.relative_to(REPO)}  ({len(rows)} patches)")

    readme = DST / "README.md"
    with open(readme, "w", encoding="utf-8") as f:
        f.write("# splits/csv_derived\n\n")
        f.write("Mesmos patches e mesmos folds de `splits/csv/` (byte-identicos aos publicados no\n")
        f.write("Zenodo) — a unica diferenca e a coluna `mask`, que aponta para `labels/subclasses/`.\n\n")
        f.write("A variante e produzida na leitura, pelo `mask_class_mapping` do\n")
        f.write("`SegmentationDataset`, ja remapeada para `0..4` (`hyperparameters.classes: 5`).\n\n")
        f.write("**A classe 0 e Water, nao NoData.** O dataset nao tem nenhum pixel NoData\n")
        f.write("(medido: 0 em 610.795.520 nas tres variantes), entao nao existe `ignore_index`.\n\n")
        f.write("## Por que derivar em vez de ler `labels/<variante>/`\n\n")
        f.write("As mascaras publicadas de `forest_nonforest` tem 8.275 pixels (em 54 dos 9.320\n")
        f.write("patches) em que Silviculture ficou como Non-forest, porque cada esquema era\n")
        f.write("rasterizado separadamente e a prioridade de sobreposicao sob `all_touched=True`\n")
        f.write("difere entre eles. Verificado sobre a v1.0.0 inteira:\n\n")
        f.write("| derivado por LUT vs publicado | pixels diferentes | patches |\n")
        f.write("|---|---|---|\n")
        f.write("| `natural_cultivated` | 0 | 0 |\n")
        f.write("| `forest_nonforest` | 8.275 | 54 |\n\n")
        f.write("Os rotulos derivados sao, por construcao, iguais aos que o pipeline corrigido\n")
        f.write("produzira na v1.1 — entao nada precisa ser retreinado quando o dataset for refeito.\n\n")
        for v, m in MASK_CLASS_MAPPING.items():
            f.write(f"## `{v}`\n\n```yaml\nn_classes: 5\nmask_class_mapping:\n")
            for k in sorted(m):
                f.write(f"  {k}: {m[k]}   # {ORIGEM[k]} -> {NOMES[v][m[k]]}\n")
            f.write("```\n\n")
    print(f"\n{total} CSVs + README -> {DST.relative_to(REPO)}")


if __name__ == "__main__":
    main()
