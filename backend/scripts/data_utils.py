"""Data utilities for loading, sampling, and splitting heterogeneous graph data."""

import torch
import numpy as np
from pathlib import Path
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader
from torch_geometric.data import HeteroData
from torch_geometric.data.storage import BaseStorage


def load_graph_and_features(device='cuda'):
    """Load heterogeneous graph and features from disk.
    
    Returns:
        HeteroData: Graph with all node features and edge indices attached
    """
    root = Path(__file__).resolve().parents[1]
    graphs_dir = root / "data" / "graphs"
    features_dir = root / "data" / "features"
    
    print("Loading graph structure...")
    # Some saved HeteroData objects include custom PyG storage classes.
    # Use torch.serialization.safe_globals to allowlist trusted globals when loading.
    with torch.serialization.safe_globals([BaseStorage]):
        data = torch.load(graphs_dir / "hetero_data_structure.pt", weights_only=False, map_location=device)

    print("Loading person features...")
    with torch.serialization.safe_globals([BaseStorage]):
        person_features = torch.load(features_dir / "person_features.pt", weights_only=False, map_location=device)
    
    # Re-attach person features to graph
    data['person'].x = person_features
    
    # Apply ToUndirected to ensure every node type can be a destination (adds reverse relations)
    try:
        from torch_geometric import transforms as T
        data = T.ToUndirected()(data)
        print("Applied ToUndirected() to graph (added reverse relations where applicable)")
    except Exception as e:
        print(f"Warning: failed to apply ToUndirected(): {e}; proceeding without it")

    print(f"Graph loaded on {device}:")
    print(f"  {data}")

    return data

def create_train_val_test_split(ratings_df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """Split ratings into train/val/test sets.
    
    Args:
        ratings_df: DataFrame with columns [userId, movieId, rating, timestamp]
        train_ratio, val_ratio, test_ratio: Split proportions
        random_seed: For reproducibility
    
    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # Temporal split (by timestamp) is more realistic; here we do random for simplicity
    n = len(ratings_df)
    indices = np.random.permutation(n)
    
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]
    
    return train_idx, val_idx, test_idx


def prepare_rating_edges(ratings_df, movie_id_to_idx, user_id_to_idx, split_indices):
    """Prepare rating edges for train/val/test.
    
    Args:
        ratings_df: DataFrame with ratings
        movie_id_to_idx: Mapping from movie ID to node index
        user_id_to_idx: Mapping from user ID to node index
        split_indices: Tuple of (train_idx, val_idx, test_idx)
    
    Returns:
        Dict with 'train', 'val', 'test' rating edges and labels
    """
    train_idx, val_idx, test_idx = split_indices
    
    def build_edges(idx):
        valid_rows = []
        ratings_list = []
        for i in idx:
            row = ratings_df.iloc[i]
            user_idx = user_id_to_idx.get(row['userId'])
            movie_idx = movie_id_to_idx.get(row['movieId'])
            rating = float(row['rating'])
            
            if user_idx is not None and movie_idx is not None:
                valid_rows.append([user_idx, movie_idx])
                ratings_list.append(rating)
        
        if valid_rows:
            edges = torch.tensor(valid_rows, dtype=torch.long).t().contiguous()
            labels = torch.tensor(ratings_list, dtype=torch.float32)
            return edges, labels
        else:
            return torch.zeros((2, 0), dtype=torch.long), torch.tensor([], dtype=torch.float32)
    
    train_edges, train_labels = build_edges(train_idx)
    val_edges, val_labels = build_edges(val_idx)
    test_edges, test_labels = build_edges(test_idx)
    
    return {
        'train': {'edges': train_edges, 'labels': train_labels},
        'val': {'edges': val_edges, 'labels': val_labels},
        'test': {'edges': test_edges, 'labels': test_labels},
    }


def create_neighbor_loader(data, edge_type, node_idx, batch_size=512, num_neighbors=[10, 10], shuffle=True):
    """Create a NeighborLoader for mini-batch sampling.
    
    Args:
        data: HeteroData graph
        edge_type: Tuple (src_type, relation, dst_type) for the target edge
        node_idx: Indices of target nodes to sample around
        batch_size: Batch size for sampling
        num_neighbors: Number of neighbors per layer (list of ints)
        shuffle: Whether to shuffle batches
    
    Returns:
        NeighborLoader for sampling
    """
    loader = NeighborLoader(
        data,
        num_neighbors={edge_type: num_neighbors},
        batch_size=batch_size,
        input_nodes=node_idx,
        shuffle=shuffle,
    )
    return loader


def prepare_link_neighbor_loaders(data, train_edges, train_labels, val_edges, val_labels, test_edges, test_labels, batch_size=512, num_neighbors=[5, 5], shuffle=True):
    """Prepare LinkNeighborLoader for train/val/test edge splits.

    Edges must be in shape [2, num_edges] and correspond to the 'user','rated','movie' relation.
    Returns three LinkNeighborLoader instances.
    """
    edge_type = ("user", "rated", "movie")

    # LinkNeighborLoader expects edge_label_index as a tuple (row, col) (or list/tuple)
    train_edge_index = (train_edges[0], train_edges[1]) if train_edges.numel() > 0 else (torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long))
    val_edge_index = (val_edges[0], val_edges[1]) if val_edges.numel() > 0 else (torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long))
    test_edge_index = (test_edges[0], test_edges[1]) if test_edges.numel() > 0 else (torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long))

    train_loader = LinkNeighborLoader(
        data,
        num_neighbors={edge_type: num_neighbors},
        edge_label_index={edge_type: train_edge_index},
        edge_label={edge_type: train_labels},
        batch_size=batch_size,
        shuffle=shuffle,
    )

    val_loader = LinkNeighborLoader(
        data,
        num_neighbors={edge_type: num_neighbors},
        edge_label_index={edge_type: val_edge_index},
        edge_label={edge_type: val_labels},
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = LinkNeighborLoader(
        data,
        num_neighbors={edge_type: num_neighbors},
        edge_label_index={edge_type: test_edge_index},
        edge_label={edge_type: test_labels},
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader


class RatingDataLoader:
    """DataLoader wrapper for rating prediction task."""
    
    def __init__(self, data, rating_edges, rating_labels, batch_size=512, shuffle=True, device='cuda'):
        """Initialize rating data loader.
        
        Args:
            data: HeteroData graph
            rating_edges: Edge indices (2, num_edges) for user->movie ratings
            rating_labels: Rating values (num_edges,)
            batch_size: Batch size
            shuffle: Whether to shuffle
            device: Device to load batches on
        """
        self.data = data
        self.rating_edges = rating_edges
        self.rating_labels = rating_labels
        self.batch_size = batch_size
        self.device = device
        
        self.num_samples = rating_edges.shape[1]
        self.indices = np.arange(self.num_samples)
        
        if shuffle:
            np.random.shuffle(self.indices)
        
        self.batch_idx = 0
    
    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size
    
    def __iter__(self):
        """Iterate over batches."""
        self.batch_idx = 0
        return self
    
    def __next__(self):
        if self.batch_idx * self.batch_size >= self.num_samples:
            raise StopIteration
        
        # Get batch indices
        start_idx = self.batch_idx * self.batch_size
        end_idx = min((self.batch_idx + 1) * self.batch_size, self.num_samples)
        batch_indices = self.indices[start_idx:end_idx]
        
        # Get batch edges and labels
        batch_edges = self.rating_edges[:, batch_indices]
        batch_labels = self.rating_labels[batch_indices]
        
        # Get user and movie node indices
        user_indices = batch_edges[0]
        movie_indices = batch_edges[1]
        
        # Get features for batch nodes
        user_features = self.data['user'].x[user_indices].to(self.device)
        movie_features = self.data['movie'].x[movie_indices].to(self.device)
        labels = batch_labels.to(self.device)
        
        self.batch_idx += 1
        
        return {
            'user_features': user_features,
            'movie_features': movie_features,
            'user_idx': user_indices.to(self.device),
            'movie_idx': movie_indices.to(self.device),
            'labels': labels,
        }
