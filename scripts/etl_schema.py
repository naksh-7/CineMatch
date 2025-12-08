"""ETL script to build a unified schema from MovieLens and IMDb raw files.

Outputs (data/processed):
- movies.csv (movie metadata merged)
- people.csv (actors/directors with nconst)
- acted_in.csv (actor nconst -> movie tconst or movieId mapping)
- directed_by.csv (director nconst -> movie mapping)
- ratings.csv (user ratings from MovieLens)
- tags.csv (user tags from MovieLens)

This is a starting point: it links MovieLens movies to IMDb via `links.csv`.
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)


def load_movielens():
    base = RAW / "movielens"
    # find movies.csv anywhere under the movielens folder (handle nested extraction)
    candidates = list(base.rglob("movies.csv"))
    if not candidates:
        raise FileNotFoundError(f"MovieLens movies.csv not found under {base}. Run download_datasets.py first.")
    movies_path = candidates[0]
    ml_dir = movies_path.parent
    links_path = ml_dir / "links.csv"
    ratings_path = ml_dir / "ratings.csv"
    tags_path = ml_dir / "tags.csv"
    if not links_path.exists() or not ratings_path.exists() or not tags_path.exists():
        raise FileNotFoundError(f"One of links/ratings/tags not found in {ml_dir}")
    movies = pd.read_csv(movies_path)
    links = pd.read_csv(links_path)
    ratings = pd.read_csv(ratings_path)
    tags = pd.read_csv(tags_path)
    return movies, links, ratings, tags


def load_imdb():
    imdb_dir = RAW / "imdb"
    # Read only needed columns to conserve memory
    basics_cols = ['tconst', 'startYear', 'runtimeMinutes', 'genres']
    names_cols = ['nconst', 'primaryName', 'birthYear', 'deathYear', 'primaryProfession']

    basics = pd.read_csv(imdb_dir / "title.basics.tsv", sep='\t', na_values='\\N', usecols=basics_cols, low_memory=True)
    names = pd.read_csv(imdb_dir / "name.basics.tsv", sep='\t', na_values='\\N', usecols=names_cols, low_memory=True)

    # principals is large; read in chunks and keep only actor/actress/director rows
    principals_cols = ['tconst', 'ordering', 'nconst', 'category', 'job', 'characters']
    principals_iter = pd.read_csv(imdb_dir / "title.principals.tsv", sep='\t', na_values='\\N', usecols=principals_cols, low_memory=True, chunksize=200000)
    principals_filtered = []
    keep_cats = set(['actor', 'actress', 'director'])
    for chunk in principals_iter:
        filt = chunk[chunk['category'].isin(keep_cats)]
        if not filt.empty:
            principals_filtered.append(filt)
    if principals_filtered:
        principals = pd.concat(principals_filtered, ignore_index=True)
    else:
        principals = pd.DataFrame(columns=principals_cols)

    # crew file is optional for current ETL; skip loading to save memory
    return basics, principals, names


def build_unified_schema():
    movies, links, ratings, tags = load_movielens()
    basics, principals, names = load_imdb()

    # normalize imdb tconst in MovieLens links (imdbId in links is like 0114709)
    links['imdbId'] = links['imdbId'].fillna('').astype(str)
    links['tconst'] = links['imdbId'].apply(lambda x: ('tt' + x.zfill(7)) if x.strip() else '')

    # merge movies with IMDb basics on tconst
    movies_with_links = movies.merge(links[['movieId', 'tconst']], on='movieId', how='left')
    merged = movies_with_links.merge(basics, on='tconst', how='left')

    # select useful columns and normalize robustly
    title_col = 'title'
    if 'title_x' in merged.columns:
        title_col = 'title_x'
    elif 'primaryTitle' in merged.columns:
        title_col = 'primaryTitle'

    genres_col = 'genres' if 'genres' in merged.columns else None

    year_col = 'startYear' if 'startYear' in merged.columns else ('year' if 'year' in merged.columns else None)

    select_cols = ['movieId', 'tconst', title_col]
    if year_col:
        select_cols.append(year_col)
    if 'runtimeMinutes' in merged.columns:
        select_cols.append('runtimeMinutes')
    if genres_col:
        select_cols.append('genres')

    movies_out = merged[select_cols].copy()
    movies_out = movies_out.rename(columns={title_col: 'title', year_col: 'year'} if year_col else {title_col: 'title'})

    # fill year from MovieLens title extraction if missing
    def extract_year_from_title(t):
        if pd.isna(t):
            return np.nan
        import re
        m = re.search(r"\((\d{4})\)$", t)
        return int(m.group(1)) if m else np.nan

    # use MovieLens title to extract year if year missing
    ml_years = movies['title'].apply(extract_year_from_title)
    movies_out['year'] = movies_out['year'].fillna(ml_years)

    # build people table from name.basics
    people = names[['nconst', 'primaryName', 'birthYear', 'deathYear', 'primaryProfession']].copy()
    people = people.rename(columns={'primaryName': 'name'})

    # principals contains cast and crew: extract actors/actresses and directors
    principals_sub = principals[principals['category'].isin(['actor', 'actress', 'director'])].copy()
    # join with tconst -> movieId mapping where available
    tconst_to_movie = links[['movieId', 'tconst']]
    principals_sub = principals_sub.merge(tconst_to_movie, on='tconst', how='left')

    acted_in = principals_sub[principals_sub['category'].isin(['actor', 'actress'])][['nconst', 'movieId', 'characters', 'ordering']].copy()
    acted_in = acted_in.rename(columns={'characters': 'role', 'ordering': 'billing_order'})

    directed = principals_sub[principals_sub['category'] == 'director'][['nconst', 'movieId']].copy()
    directed = directed.rename(columns={'nconst': 'director_nconst'})

    # save outputs
    movies_out.to_csv(PROCESSED / 'movies.csv', index=False)
    people.to_csv(PROCESSED / 'people.csv', index=False)
    acted_in.to_csv(PROCESSED / 'acted_in.csv', index=False)
    directed.to_csv(PROCESSED / 'directed_by.csv', index=False)
    ratings.to_csv(PROCESSED / 'ratings.csv', index=False)
    tags.to_csv(PROCESSED / 'tags.csv', index=False)

    print('Processed files written to', PROCESSED)


if __name__ == '__main__':
    build_unified_schema()