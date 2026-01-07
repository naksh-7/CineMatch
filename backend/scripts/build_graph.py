"""GPU-optimized heterogeneous graph construction using vectorized operations.

Key optimizations:
1. Vectorized edge construction (pandas.map instead of iterrows)
2. Separate storage of large feature tensors to avoid CPU OOM
3. GPU acceleration for edge index construction

Outputs:
- data/graphs/hetero_data_structure.pt: Graph structure (edges + small features)
- data/features/actor_features.pt: Large actor/person features (saved separately)
- data/graphs/graph_metadata.json: Graph statistics and schema
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
FEATURES = ROOT / "data" / "features"
GRAPHS = ROOT / "data" / "graphs"
GRAPHS.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")


def load_processed_data():
    """Load all processed CSVs."""
    print("\n=== Loading Processed Data ===")
    movies = pd.read_csv(PROCESSED / "movies.csv")
    acted_in = pd.read_csv(PROCESSED / "acted_in.csv")
    directed_by = pd.read_csv(PROCESSED / "directed_by.csv")
    ratings = pd.read_csv(PROCESSED / "ratings.csv")
    tags = pd.read_csv(PROCESSED / "tags.csv")
    
    # Keep only people who appear in acted_in or directed_by
    people_in_graph = set(acted_in['nconst'].unique()) | set(directed_by['director_nconst'].unique())
    people = pd.read_csv(PROCESSED / "people.csv")
    people = people[people['nconst'].isin(people_in_graph)].reset_index(drop=True)
    
    print(f"  Movies: {len(movies)}")
    print(f"  People: {len(people)}")
    print(f"  Acted-in edges: {len(acted_in)}")
    print(f"  Directed-by edges: {len(directed_by)}")
    print(f"  Ratings: {len(ratings)}")
    print(f"  Tags: {len(tags)}")
    
    return movies, people, acted_in, directed_by, ratings, tags


def build_id_mappings(movies, people, ratings, tags):
    """Create ID mappings for all entity types."""
    print("\n=== Building ID Mappings ===")
    
    movie_id_to_idx = {mid: idx for idx, mid in enumerate(movies['movieId'].values)}
    person_nconst_to_idx = {nconst: idx for idx, nconst in enumerate(people['nconst'].values)}
    unique_users = sorted(ratings['userId'].unique())
    user_id_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
    unique_tags = sorted([str(t) for t in tags['tag'].dropna().unique()])
    tag_to_idx = {tag: idx for idx, tag in enumerate(unique_tags)}
    
    print(f"  Movies: {len(movie_id_to_idx)}")
    print(f"  People: {len(person_nconst_to_idx)}")
    print(f"  Users: {len(user_id_to_idx)}")
    print(f"  Tags: {len(unique_tags)}")
    
    return movie_id_to_idx, person_nconst_to_idx, user_id_to_idx, tag_to_idx, unique_users, unique_tags


def load_movie_features():
    """Load precomputed movie embeddings."""
    print("\n=== Loading Movie Features ===")
    features = np.load(FEATURES / "movie_features.npy")
    print(f"  Shape: {features.shape}")
    tensor = torch.from_numpy(features).float()
    return tensor


def build_node_features(movies, people, ratings, tags, movie_features, unique_users):
    """Build node feature tensors (efficiently)."""
    print("\n=== Building Node Features ===")
    
    # Movie features: precomputed embeddings
    movie_x = movie_features
    print(f"  Movie features: {movie_x.shape}")
    
    # Person/Actor features: random initialization (small dim to fit in GPU)
    # We'll save this separately to avoid CPU OOM
    n_people = len(people)
    people_x = torch.randn(n_people, 128).float() * 0.1
    print(f"  Actor features (random init): {people_x.shape}")
    
    # User features: aggregated from rated movies (vectorized)
    print(f"  Aggregating user features from ratings...")
    user_id_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
    movie_id_to_idx = {mid: idx for idx, mid in enumerate(movies['movieId'].values)}
    
    user_x = torch.zeros(len(unique_users), movie_x.shape[1]).float()
    
    # Vectorized: for each user, find mean embedding of rated movies
    for user_idx, user_id in enumerate(tqdm(unique_users, desc="Aggregating user features", leave=False)):
        user_movie_ids = ratings[ratings['userId'] == user_id]['movieId'].values
        valid_indices = np.array([movie_id_to_idx[mid] for mid in user_movie_ids if mid in movie_id_to_idx])
        if len(valid_indices) > 0:
            user_x[user_idx] = movie_x[valid_indices].mean(dim=0)
    
    print(f"  User features (aggregated): {user_x.shape}")
    
    # Tag features: random initialization
    n_tags = len(tags['tag'].unique())
    tag_x = torch.randn(n_tags, 64).float() * 0.1
    print(f"  Tag features (random init): {tag_x.shape}")
    
    return movie_x, people_x, user_x, tag_x, user_id_to_idx


def build_edges_vectorized(movies, people, acted_in, directed_by, ratings, tags, movie_id_to_idx, person_nconst_to_idx, user_id_to_idx, tag_to_idx):
    """Build edges using vectorized pandas operations (no iterrows)."""
    print("\n=== Building Edges (Vectorized) ===")
    
    edges = {}
    
    # --- ACTED_IN: person -> movie ---
    print("Building ACTED_IN edges (vectorized)...")
    acted_in_copy = acted_in.copy()
    acted_in_copy['person_idx'] = acted_in_copy['nconst'].map(person_nconst_to_idx)
    acted_in_copy['movie_idx'] = acted_in_copy['movieId'].map(movie_id_to_idx)
    valid = acted_in_copy.dropna(subset=['person_idx', 'movie_idx'])
    if len(valid) > 0:
        src = valid['person_idx'].astype(int).values
        dst = valid['movie_idx'].astype(int).values
        edges[('person', 'acted_in', 'movie')] = torch.tensor([src, dst], dtype=torch.long)
        print(f"  ACTED_IN: {len(src)} edges")
    
    # --- DIRECTED_BY: person -> movie ---
    print("Building DIRECTED_BY edges (vectorized)...")
    directed_by_copy = directed_by.copy()
    directed_by_copy['person_idx'] = directed_by_copy['director_nconst'].map(person_nconst_to_idx)
    directed_by_copy['movie_idx'] = directed_by_copy['movieId'].map(movie_id_to_idx)
    valid = directed_by_copy.dropna(subset=['person_idx', 'movie_idx'])
    if len(valid) > 0:
        src = valid['person_idx'].astype(int).values
        dst = valid['movie_idx'].astype(int).values
        edges[('person', 'directed_by', 'movie')] = torch.tensor([src, dst], dtype=torch.long)
        print(f"  DIRECTED_BY: {len(src)} edges")
    
    # --- RATED_BY: user -> movie ---
    print("Building RATED_BY edges (vectorized)...")
    ratings_copy = ratings.copy()
    ratings_copy['user_idx'] = ratings_copy['userId'].map(user_id_to_idx)
    ratings_copy['movie_idx'] = ratings_copy['movieId'].map(movie_id_to_idx)
    valid = ratings_copy.dropna(subset=['user_idx', 'movie_idx'])
    if len(valid) > 0:
        src = valid['user_idx'].astype(int).values
        dst = valid['movie_idx'].astype(int).values
        edges[('user', 'rated', 'movie')] = torch.tensor([src, dst], dtype=torch.long)
        print(f"  RATED_BY: {len(src)} edges")
    
    # --- TAGGED_BY: user -> movie ---
    print("Building TAGGED_BY edges (vectorized)...")
    tags_copy = tags.copy()
    tags_copy['user_idx'] = tags_copy['userId'].map(user_id_to_idx)
    tags_copy['movie_idx'] = tags_copy['movieId'].map(movie_id_to_idx)
    valid = tags_copy.dropna(subset=['user_idx', 'movie_idx'])
    if len(valid) > 0:
        src = valid['user_idx'].astype(int).values
        dst = valid['movie_idx'].astype(int).values
        edges[('user', 'tagged', 'movie')] = torch.tensor([src, dst], dtype=torch.long)
        print(f"  TAGGED_BY: {len(src)} edges")
    
    return edges


def build_hetero_graph(movie_x, people_x, user_x, tag_x, edges, n_people, n_users, n_tags):
    """Construct HeteroData object."""
    print("\n=== Building HeteroData ===")
    
    data = HeteroData()
    
    # Add node features
    data['movie'].x = movie_x
    data['person'].num_nodes = n_people
    data['user'].x = user_x
    data['tag'].x = tag_x
    
    # Add edges
    for (src_type, rel, dst_type), edge_idx in edges.items():
        data[src_type, rel, dst_type].edge_index = edge_idx
    
    print(f"\nGraph Summary:")
    print(f"  {data}")
    
    return data


def save_graph_and_metadata(data, people_x):
    """Save graph structure and large features separately to avoid CPU OOM."""
    print("\n=== Saving Graph ===")
    
    # 1. Remove large person features from data to avoid CPU copy during serialization
    # Store only node count
    person_num_nodes = data['person'].num_nodes
    
    # 2. Save the graph structure (without person features)
    print(f"Saving graph structure (edges + small features) to {GRAPHS / 'hetero_data_structure.pt'}...")
    torch.save(data, GRAPHS / "hetero_data_structure.pt")
    
    # 3. Save large person features separately (to features directory)
    print(f"Saving large person features separately to {FEATURES / 'person_features.pt'}...")
    torch.save(people_x, FEATURES / "person_features.pt")
    
    # 4. Generate and save metadata
    metadata = {
        "num_nodes": {
            "movie": data['movie'].num_nodes,
            "person": person_num_nodes,
            "user": data['user'].num_nodes,
            "tag": data['tag'].num_nodes,
        },
        "num_edges": {
            f"{src}-{rel}-{dst}": data[src, rel, dst].edge_index.shape[1]
            for src, rel, dst in data.edge_types
        },
        "feature_dims": {
            "movie": data['movie'].x.shape[1],
            "person": people_x.shape[1],
            "user": data['user'].x.shape[1],
            "tag": data['tag'].x.shape[1],
        },
        "edge_types": [f"{src}-{rel}-{dst}" for src, rel, dst in data.edge_types],
        "node_types": data.node_types,
        "loading_instructions": {
            "structure": "torch.load(GRAPHS / 'hetero_data_structure.pt')",
            "person_features": "torch.load(FEATURES / 'person_features.pt')",
            "reattach": "data['person'].x = person_features.to(device)",
        },
    }
    
    with open(GRAPHS / "graph_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 70)
    print("Graph Metadata")
    print("=" * 70)
    print(json.dumps(metadata, indent=2))
    
    return metadata


def main():
    print("=" * 70)
    print("GPU-Optimized Heterogeneous Graph Construction (Vectorized)")
    print("=" * 70)
    
    # Load data
    movies, people, acted_in, directed_by, ratings, tags = load_processed_data()
    
    # Build mappings
    movie_id_to_idx, person_nconst_to_idx, user_id_to_idx, tag_to_idx, unique_users, unique_tags = \
        build_id_mappings(movies, people, ratings, tags)
    
    # Load features
    movie_x = load_movie_features()
    
    # Build node features
    movie_x, people_x, user_x, tag_x, _ = build_node_features(
        movies, people, ratings, tags, movie_x, unique_users
    )
    
    # Build edges (vectorized, no iterrows)
    edges = build_edges_vectorized(
        movies, people, acted_in, directed_by, ratings, tags,
        movie_id_to_idx, person_nconst_to_idx, user_id_to_idx, tag_to_idx
    )
    
    # Build HeteroData
    data = build_hetero_graph(
        movie_x, people_x, user_x, tag_x, edges,
        len(people), len(unique_users), len(unique_tags)
    )
    
    # Save graph and metadata (person features saved separately)
    metadata = save_graph_and_metadata(data, people_x)
    
    print("\n" + "=" * 70)
    print("Graph Construction Complete!")
    print("=" * 70)
    print("\nTo load the graph in training:")
    print("  data = torch.load('data/graphs/hetero_data_structure.pt')")
    print("  person_x = torch.load('data/features/person_features.pt')")
    print("  data['person'].x = person_x.to(device)")


if __name__ == '__main__':
    main()
