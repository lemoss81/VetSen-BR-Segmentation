# VetSen-BR — `forest_nonforest`

Physiognomic labeling: vegetation is split by **physiognomy** (tree-form vs
non-tree-form), following the ET-EDGV 3.0 interpretation of the Army topographic
charts.

Masks are single-band `uint8` GeoTIFFs, 256 × 256 px, aligned to `images/` by filename.

## Pixel value → class

| Value | Class | Notes |
|-------|-------|-------|
| 0 | NoData | ignored in training/evaluation |
| 1 | Water | |
| 2 | Built-up | |
| 3 | Forest | tree-form (forest) vegetation |
| 4 | Non-forest | grassland, savanna and other non-tree-form vegetation |
| 5 | Barren Land | exposed soil / bare surfaces |

5 scored classes (1–5); class 0 is masked out (`ignore_index=0`).

Pair each mask with the 10-band image of the same name in `../../images/`.
See the root `README.md` for imagery, splits and citation.
