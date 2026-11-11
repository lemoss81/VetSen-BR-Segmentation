# VetSen-BR — `natural_cultivated`

Anthropization labeling: vegetation is split by **land use / anthropization**
(natural vs cultivated), rather than by physiognomy. Same patches and same
Water/Built-up/Barren Land classes as the other variants — only the two vegetation
classes differ.

Masks are single-band `uint8` GeoTIFFs, 256 × 256 px, aligned to `images/` by filename.

## Pixel value → class

| Value | Class | Notes |
|-------|-------|-------|
| 0 | NoData | ignored in training/evaluation |
| 1 | Water | |
| 2 | Built-up | |
| 3 | Natural | natural (non-anthropic) vegetation |
| 4 | Cultivated | cultivated / managed vegetation (e.g. cropland, silviculture) |
| 5 | Barren Land | exposed soil / bare surfaces |

5 scored classes (1–5); class 0 is masked out (`ignore_index=0`).

Pair each mask with the 10-band image of the same name in `../../images/`.
See the root `README.md` for imagery, splits and citation.
