# VetSen-BR

**Sentinel-2 semantic-segmentation benchmark for Brazil, labeled from official Army topographic charts (ET-EDGV 3.0).**

VetSen-BR pairs 10-band Sentinel-2 imagery with land-cover segmentation masks derived
from official Brazilian topographic vector cartography (BDGEx, structured under
ET-EDGV 3.0). It is built for training and benchmarking semantic-segmentation models
across the Brazilian territory.

## At a glance

| | |
|---|---|
| Patches | **9,320** non-overlapping tiles (~61,080 km²) |
| Tile size | 256 × 256 px, 10 m |
| Imagery | 10-band Sentinel-2, `uint16` GeoTIFF |
| Labels | single-band `uint8` GeoTIFF; `0` = NoData (ignored) |
| Biomes | Atlantic Forest (47.8%), Cerrado (29.8%), Amazon (12.8%), Pampa (7.4%), Pantanal (2.2%) |
| Vegetation share | ~94% of the labeled area |
| License | MIT |

## Imagery

Each image is a 10-band Sentinel-2 surface-reflectance stack, all bands resampled to
10 m. Band order (channel 1 → 10):

| Ch | Band | Name | Native |
|----|------|------|-----------|
| 1 | B02 | Blue | 10 m |
| 2 | B03 | Green | 10 m |
| 3 | B04 | Red | 10 m |
| 4 | B08 | NIR | 10 m |
| 5 | B05 | Red-edge 1 | 20 m → 10 m |
| 6 | B06 | Red-edge 2 | 20 m → 10 m |
| 7 | B07 | Red-edge 3 | 20 m → 10 m |
| 8 | B8A | Red-edge 4 / narrow NIR | 20 m → 10 m |
| 9 | B11 | SWIR 1 | 20 m → 10 m |
| 10 | B12 | SWIR 2 | 20 m → 10 m |

Cloud- and shadow-affected patches were removed in a two-stage quality control
(Sentinel-2 Scene Classification Layer, then the OmniCloudMask deep-learning model),
followed by visual inspection by two analysts. The final dataset is cloud-free; there
is no cloud class.

## Label variants ("one image set, several labelings")

Every variant shares the **same images** — only the label scheme changes. Pick one
`labels/<variant>/` folder and pair each mask with the image of the same filename in
`images/`.

| Folder | Scheme | Classes |
|--------|--------|---------|
| `forest_nonforest` | Physiognomic (tree vs non-tree form) | Water, Built-up, Forest, Non-forest, Barren Land |
| `natural_cultivated` | Anthropization (natural vs cultivated) | Water, Built-up, Natural, Cultivated, Barren Land |
| `subclasses` | Fine-grained (10 classes) | Water, Built-up, Natural forest, Savanna, Grassland, Wetland, Silviculture, Cropland, Bare soil, Rock |

See each folder's `README.md` for the exact pixel-value → class mapping.

## Class definitions (ET-EDGV 3.0)

Labels follow the land-cover feature classes of the Brazilian Army's **ET-EDGV 3.0**
technical specification; the vegetation definitions, in turn, derive from the IBGE
*Manual Técnico da Vegetação Brasileira*. The definitions below are condensed and
translated from the ET-EDGV 3.0 Dictionary of Classes (original EDGV feature name in
parentheses).

| Class | Definition (ET-EDGV 3.0) |
|-------|--------------------------|
| **Water** (*Massa d'Água*) | A water body represented as a polygon: oceans, bays, rivers, coves, oxbow lakes, lakes, ponds, lagoons and reservoirs/dams. |
| **Built-up** (*Área Densamente Edificada / Área Construída*) | An effectively built-up surface — a set of buildings, architectural/engineering works and general permanent, adjacent constructions, including roads. Where structures are too close to be mapped individually, the outline of the whole built-up block is taken. |
| **Natural forest** (*Floresta*) | Vegetation of tall trees (exceeding 40 m in Amazonia), locally with palms, lianas and bamboo among the tree individuals; outside Amazonia some types are seasonally deciduous. |
| **Savanna** (*Cerrado*) | Two-stratum vegetation — a grassy layer and a tree layer of gnarled, irregularly branched individuals; comprises four physiognomies (forested/*Cerradão*, wooded/*Campo Cerrado*, park and grassy). |
| **Grassland** (*Campo*) | A predominantly graminoid (grass–woody) physiognomy, spanning the grassy forms of Campinarana, Savanna (Cerrado), Steppe-Savanna (Caatinga) and Steppe (southern Brazilian grasslands). |
| **Wetland** (*Brejo / Pântano*) | A pioneer formation under fluvial and/or lacustrine influence on alluvial plains, reflecting river floods in the rainy season or yearly waterlogged depressions. |
| **Silviculture** (*Reflorestamento*) | An area where a forest was planted on naturally forested land that had lost its original characteristics through human or natural action. |
| **Cropland** (*Vegetação Cultivada*) | Vegetation of cultivated plant species grown for ecological, food, industrial or soil-protection (anti-erosion) purposes. |
| **Bare soil** (*Terreno Exposto*) | A site of exposed soil, characterised by the predominant absence of vegetation (any scattered vegetation being negligible in the area's context). |
| **Rock** (*Rocha*) | A natural formation of aggregated minerals resulting from a geological process. |

**Aggregation into the coarser variants.** Water, Built-up and **Barren Land = Bare soil + Rock**
are identical across variants; only the vegetation subclasses are regrouped:

- `forest_nonforest` (physiognomy): **Forest** = Natural forest + Savanna + Silviculture;
  **Non-forest** = Grassland + Wetland + Cropland.
- `natural_cultivated` (anthropization): **Natural** = Natural forest + Savanna + Grassland + Wetland;
  **Cultivated** = Silviculture + Cropland.

*Source: Brazilian Army, Diretoria de Serviço Geográfico (DSG). Especificação Técnica
para Estruturação de Dados Geoespaciais Vetoriais (ET-EDGV), v. 3.0 — Dictionary of
Classes. Vegetation definitions after IBGE, Manual Técnico da Vegetação Brasileira,
2nd ed., Rio de Janeiro: IBGE, 2012.*

## Directory layout

```
VetSen-BR/
├── images/                       # 10-band Sentinel-2 patches (shared by all variants)
├── labels/
│   ├── forest_nonforest/         # masks + README.md
│   ├── natural_cultivated/       # masks + README.md
│   └── subclasses/               # masks + README.md
└── splits/                       # chart-blocked 5-fold cross-validation
    ├── catalog.csv               # per-chart metadata (biome, image date, n_patches, fold)
    ├── folds_por_carta.csv       # chart → fold
    ├── folds_por_patch.csv       # patch → fold
    └── csv/<variant>/fold{0..4}/{train,val,test}.csv
```

Image and mask filenames are identical, so `images/<name>.tif` ↔
`labels/<variant>/<name>.tif`. Filename pattern:
`VetSenBR_s<chart>_a<img>_r<row>_c<col>.tif`.

## Splits (no spatial leakage)

Train/validation/test folds use **StratifiedGroupKFold grouped by topographic chart**:
all patches from a single chart stay in the same fold, so adjacent patches never leak
between training and test. Folds are stratified by biome (~20% of the data each) and
balanced by patch count. The minority classes (Water, Built-up, Barren Land) appear in
all five folds; the Pantanal biome (3 charts, 209 patches) appears in three folds and
is absent from two. In `csv/<variant>/fold<f>/`, the files `train.csv`, `val.csv` and
`test.csv` list `image,mask` paths relative to the dataset root.

## Code & citation

Training and benchmarking code: <https://github.com/lemoss81/VetSen-BR-Segmentation>

```bibtex
@misc{vetsenbr,
  title  = {VetSen-BR: a Sentinel-2 semantic-segmentation benchmark for Brazil},
  author = {Mateus L. de Abrantes and André L. Favalessa and Maurício C. M. de Paulo and Felipe Ferrari and Raphael L. F. Greco and Philipe Borba and Daniel R. dos Santos},
  year   = {2026},
  note   = {https://www.kaggle.com/datasets/datasetbr1/vetsen-br}
}
```
