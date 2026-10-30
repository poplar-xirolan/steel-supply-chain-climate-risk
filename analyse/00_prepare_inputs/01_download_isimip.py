"""Download the complete ISIMIP input collections used by the paper."""

from __future__ import annotations

from pathlib import Path as _DataPath
import sys as _data_sys
_ANALYSE_ROOT = next(p for p in _DataPath(__file__).resolve().parents if (p / 'run_analyse.py').is_file())
if str(_ANALYSE_ROOT / 'library') not in _data_sys.path:
    _data_sys.path.insert(0, str(_ANALYSE_ROOT / 'library'))
from steel_supply.pipeline_paths import data_path, science_path, bundle_root


import argparse
import concurrent.futures
import shutil
import urllib.request
from pathlib import Path
from urllib.parse import urlparse


ROOT = bundle_root()
MANIFESTS = {
    "tasmax": "tasmax_urls.txt",
    "tas": "tas_urls.txt",
    "hurs": "hurs_urls.txt",
    "wind": "tropical_cyclone_urls.txt",
    "wildfire": "wildfire_urls.txt",
}


def download_one(url: str, output: Path, overwrite: bool) -> str:
    destination = output / Path(urlparse(url).path).name
    if destination.exists() and not overwrite:
        return "skipped"
    partial = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url) as response, partial.open("wb") as target:
        shutil.copyfileobj(response, target)
    partial.replace(destination)
    return "downloaded"


def read_urls(dataset: str) -> list[str]:
    manifest = (
        data_path('input/isimip_download_lists')
        / MANIFESTS[dataset]
    )
    return [
        line.strip()
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["all", *MANIFESTS],
        default="all",
        help="Download one variable/hazard collection or all collections.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=data_path('input/isimip'),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    datasets = list(MANIFESTS) if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        urls = read_urls(dataset)
        output = args.output_root / dataset
        output.mkdir(parents=True, exist_ok=True)
        print(f"{dataset}: downloading {len(urls)} files to {output}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            status = list(
                pool.map(
                    lambda url: download_one(url, output, args.overwrite),
                    urls,
                )
            )
        print({name: status.count(name) for name in sorted(set(status))})


if __name__ == "__main__":
    main()
