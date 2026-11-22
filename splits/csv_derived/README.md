# splits/csv_derived

Mesmos patches e mesmos folds de `splits/csv/` (byte-identicos aos publicados no
Zenodo) — a unica diferenca e a coluna `mask`, que aponta para `labels/subclasses/`.

A variante e produzida na leitura, pelo `mask_class_mapping` do
`SegmentationDataset`, ja remapeada para `0..4` (`hyperparameters.classes: 5`).

**A classe 0 e Water, nao NoData.** O dataset nao tem nenhum pixel NoData
(medido: 0 em 610.795.520 nas tres variantes), entao nao existe `ignore_index`.

## Por que derivar em vez de ler `labels/<variante>/`

As mascaras publicadas de `forest_nonforest` tem 8.275 pixels (em 54 dos 9.320
patches) em que Silviculture ficou como Non-forest, porque cada esquema era
rasterizado separadamente e a prioridade de sobreposicao sob `all_touched=True`
difere entre eles. Verificado sobre a v1.0.0 inteira:

| derivado por LUT vs publicado | pixels diferentes | patches |
|---|---|---|
| `natural_cultivated` | 0 | 0 |
| `forest_nonforest` | 8.275 | 54 |

Os rotulos derivados sao, por construcao, iguais aos que o pipeline corrigido
produzira na v1.1 — entao nada precisa ser retreinado quando o dataset for refeito.

## `forest_nonforest`

```yaml
n_classes: 5
mask_class_mapping:
  1: 0   # Water -> Water
  2: 1   # Built-up -> Built-up
  3: 2   # Natural forest -> Forest
  4: 2   # Savanna -> Forest
  5: 3   # Grassland -> Non-forest
  6: 3   # Wetland -> Non-forest
  7: 2   # Silviculture -> Forest
  8: 3   # Cropland -> Non-forest
  9: 4   # Bare soil -> Barren Land
  10: 4   # Rock -> Barren Land
```

## `natural_cultivated`

```yaml
n_classes: 5
mask_class_mapping:
  1: 0   # Water -> Water
  2: 1   # Built-up -> Built-up
  3: 2   # Natural forest -> Natural
  4: 2   # Savanna -> Natural
  5: 2   # Grassland -> Natural
  6: 2   # Wetland -> Natural
  7: 3   # Silviculture -> Cultivated
  8: 3   # Cropland -> Cultivated
  9: 4   # Bare soil -> Barren Land
  10: 4   # Rock -> Barren Land
```

