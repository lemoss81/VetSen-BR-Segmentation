# VetSen-BR — `subclasses`

Fine-grained labeling with 10 land-cover classes, giving the detailed breakdown behind
the aggregated `forest_nonforest` and `natural_cultivated` variants.

Masks are single-band `uint8` GeoTIFFs, 256 × 256 px, aligned to `images/` by filename.

## Pixel value → class

| Value | Class |
|-------|-------|
| 0 | NoData (ignored) |
| 1 | Water |
| 2 | Built-up |
| 3 | Natural forest |
| 4 | Savanna |
| 5 | Grassland |
| 6 | Wetland |
| 7 | Silviculture |
| 8 | Cropland |
| 9 | Bare soil |
| 10 | Rock |

10 scored classes (1–10); class 0 is masked out (`ignore_index=0`).

## Relation to the aggregated variants

- **forest_nonforest:** Forest = {Natural forest, Savanna, Silviculture}; Non-forest =
  {Grassland, Wetland, Cropland}; Barren Land = {Bare soil, Rock}.
- **natural_cultivated:** Natural = {Natural forest, Savanna, Grassland, Wetland};
  Cultivated = {Silviculture, Cropland}; Barren Land = {Bare soil, Rock}.

Pair each mask with the 10-band image of the same name in `../../images/`.
See the root `README.md` for imagery, splits and citation.
