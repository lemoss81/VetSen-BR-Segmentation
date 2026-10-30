import argparse
import csv
import json
from pathlib import Path

import numpy as np
import rasterio


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute per-band statistics for a raster dataset indexed by CSV."
    )
    parser.add_argument("--csv", required=True, help="Path to CSV with image column.")
    parser.add_argument(
        "--root-dir",
        default=None,
        help="Optional root dir prepended to relative paths in the CSV.",
    )
    parser.add_argument(
        "--image-column",
        default="image",
        help="CSV column containing image paths.",
    )
    parser.add_argument(
        "--bands",
        default=None,
        help="Comma-separated 1-based band indices. Default: all bands.",
    )
    parser.add_argument(
        "--scale-max",
        type=float,
        default=65535.0,
        help="Value used to report normalized mean/std in [0,1].",
    )
    parser.add_argument(
        "--ignore-zero",
        action="store_true",
        help="Ignore zero-valued pixels in addition to masked/nodata pixels.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to save stats as JSON.",
    )
    return parser.parse_args()


def resolve_path(root_dir, path_str):
    path = Path(path_str)
    if path.is_absolute() or root_dir is None:
        return path
    return Path(root_dir) / path_str


def compute_stats(csv_path, root_dir=None, image_column="image", bands=None, ignore_zero=False):
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError("CSV is empty.")

    first_image_path = resolve_path(root_dir, rows[0][image_column])
    with rasterio.open(first_image_path) as src:
        band_indexes = list(range(1, src.count + 1)) if bands is None else list(bands)

    n_bands = len(band_indexes)
    mins = np.full(n_bands, np.inf, dtype=np.float64)
    maxs = np.full(n_bands, -np.inf, dtype=np.float64)
    sums = np.zeros(n_bands, dtype=np.float64)
    sumsqs = np.zeros(n_bands, dtype=np.float64)
    counts = np.zeros(n_bands, dtype=np.int64)

    for idx, row in enumerate(rows, start=1):
        image_path = resolve_path(root_dir, row[image_column])
        with rasterio.open(image_path) as src:
            data = src.read(band_indexes, masked=True)

        for band_pos in range(n_bands):
            band = data[band_pos]
            valid = band.compressed() if np.ma.isMaskedArray(band) else np.asarray(band).ravel()
            if ignore_zero:
                valid = valid[valid != 0]
            if valid.size == 0:
                continue
            valid = valid.astype(np.float64, copy=False)
            mins[band_pos] = min(mins[band_pos], valid.min())
            maxs[band_pos] = max(maxs[band_pos], valid.max())
            sums[band_pos] += valid.sum()
            sumsqs[band_pos] += np.square(valid).sum()
            counts[band_pos] += valid.size

        if idx % 500 == 0:
            print(f"Processed {idx}/{len(rows)} images...")

    means = sums / counts
    variances = (sumsqs / counts) - np.square(means)
    variances = np.maximum(variances, 0.0)
    stds = np.sqrt(variances)

    result = {
        "csv_path": str(csv_path),
        "root_dir": None if root_dir is None else str(root_dir),
        "image_column": image_column,
        "band_indexes": band_indexes,
        "num_images": len(rows),
        "ignore_zero": ignore_zero,
        "bands": [],
    }
    for i, band_index in enumerate(band_indexes):
        result["bands"].append(
            {
                "band_index": band_index,
                "count": int(counts[i]),
                "min": float(mins[i]),
                "max": float(maxs[i]),
                "mean_raw": float(means[i]),
                "std_raw": float(stds[i]),
            }
        )
    return result


def add_scaled_stats(result, scale_max):
    for band in result["bands"]:
        band["mean_scaled"] = band["mean_raw"] / scale_max
        band["std_scaled"] = band["std_raw"] / scale_max
    result["scale_max"] = scale_max
    result["mean_scaled_list"] = [band["mean_scaled"] for band in result["bands"]]
    result["std_scaled_list"] = [band["std_scaled"] for band in result["bands"]]
    return result


def main():
    args = parse_args()
    bands = None if args.bands is None else [int(x.strip()) for x in args.bands.split(",") if x.strip()]
    csv_path = Path(args.csv)
    root_dir = None if args.root_dir is None else Path(args.root_dir)

    result = compute_stats(
        csv_path=csv_path,
        root_dir=root_dir,
        image_column=args.image_column,
        bands=bands,
        ignore_zero=args.ignore_zero,
    )
    result = add_scaled_stats(result, args.scale_max)

    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved JSON to {output_path}")


if __name__ == "__main__":
    main()
