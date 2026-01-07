"""Download MovieLens and IMDb public datasets and save under data/raw.

MovieLens (GroupLens): https://files.grouplens.org/datasets/movielens/ml-latest.zip
IMDb public datasets: https://datasets.imdbws.com/

This script downloads and extracts these datasets into `data/raw`.
"""
import os
import sys
import requests
import zipfile
import gzip
import shutil
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)


def download(url: str, dest: Path, chunk_size=8192):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"Already downloaded: {dest}")
        return
    print(f"Downloading {url} -> {dest}")
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit='B', unit_scale=True) as pbar:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def download_movielens():
    url = "https://files.grouplens.org/datasets/movielens/ml-latest.zip"
    dest = RAW / "movielens" / "ml-latest.zip"
    download(url, dest)
    extract_dir = RAW / "movielens" / "ml-latest"
    if not extract_dir.exists():
        print(f"Extracting {dest} -> {extract_dir}")
        with zipfile.ZipFile(dest, 'r') as z:
            z.extractall(extract_dir)
    else:
        print(f"Already extracted: {extract_dir}")


IMDB_FILES = [
    "title.basics.tsv.gz",
    "title.principals.tsv.gz",
    "title.crew.tsv.gz",
    "name.basics.tsv.gz",
]


def download_imdb():
    base = "https://datasets.imdbws.com"
    imdb_dir = RAW / "imdb"
    imdb_dir.mkdir(parents=True, exist_ok=True)
    for fname in IMDB_FILES:
        url = f"{base}/{fname}"
        dest = imdb_dir / fname
        download(url, dest)
        # decompress .gz to .tsv if not yet present
        out_tsv = imdb_dir / fname.replace('.gz', '')
        if not out_tsv.exists():
            print(f"Decompressing {dest} -> {out_tsv}")
            with gzip.open(dest, 'rb') as f_in, open(out_tsv, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        else:
            print(f"Already decompressed: {out_tsv}")


def main():
    download_movielens()
    download_imdb()
    print("Downloads complete. Raw files are in:")
    for p in (RAW / 'movielens').glob('*'):
        print(' -', p)
    for p in (RAW / 'imdb').glob('*'):
        print(' -', p)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('Error during download:', e)
        sys.exit(1)
