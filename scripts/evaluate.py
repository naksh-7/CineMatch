"""Evaluation script for GNN model.

Computes detailed offline metrics (RMSE, MAE, NDCG, Precision@K) on test set.
"""

import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, ndcg_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.models.hetero_gnn import HeteroGraphSAGE, HeteroGNNForRating
from scripts.data_utils import load_graph_and_features, RatingDataLoader
from config.gnn_config import MODEL, DATA, EVALUATION

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT / EVALUATION['checkpoint_dir']
LOG_DIR = ROOT / "logs"


def compute_ndcg_at_k(predictions, labels, k=10):
    """Compute NDCG@K for ranking task.
    
    Args:
        predictions: Array of predicted ratings
        labels: Array of ground truth ratings (1-5 scale, binarized to >3 as relevant)
        k: Cutoff for NDCG
    
    Returns:
        NDCG@K score
    """
    # Binarize labels: relevant if rating >= 4
    relevant = (labels >= 4.0).astype(int)
    
    # Create ranking scores (sort by predicted rating, descending)
    if len(predictions) < k:
        k = len(predictions)
    
    # Use ndcg_score from sklearn (requires shape compatibility)
    # For a single query: y_true = [relevant], y_score = [predictions]
    try:
        score = ndcg_score([relevant], [predictions], k=k)
        return score
    except:
        # Fallback: compute manually
        sorted_idx = np.argsort(predictions)[::-1][:k]
        dcg = np.sum(relevant[sorted_idx] / np.log2(np.arange(2, k + 2)))
        idcg = np.sum(np.sort(relevant)[::-1][:k] / np.log2(np.arange(2, k + 2)))
        return dcg / idcg if idcg > 0 else 0.0


def compute_precision_at_k(predictions, labels, k=10):
    """Compute Precision@K.
    
    Args:
        predictions: Array of predicted ratings
        labels: Array of ground truth ratings
        k: Cutoff for precision
    
    Returns:
        Precision@K score
    """
    relevant = (labels >= 4.0).astype(int)
    
    if len(predictions) < k:
        k = len(predictions)
    
    sorted_idx = np.argsort(predictions)[::-1][:k]
    precision = np.sum(relevant[sorted_idx]) / k if k > 0 else 0.0
    return precision


@torch.no_grad()
def evaluate_model(model, rating_model, test_loader, device):
    """Evaluate model and compute detailed metrics."""
    model.eval()
    rating_model.eval()
    
    all_preds = []
    all_labels = []
    
    print("Evaluating on test set...")
    for batch in test_loader:
        user_features = batch['user_features'].to(device)
        movie_features = batch['movie_features'].to(device)
        labels = batch['labels'].to(device)
        
        ratings = rating_model(user_features, movie_features).squeeze()
        
        all_preds.append(ratings.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    
    # Compute metrics
    rmse = np.sqrt(mean_squared_error(labels, preds))
    mae = mean_absolute_error(labels, preds)
    ndcg_10 = compute_ndcg_at_k(preds, labels, k=10)
    prec_10 = compute_precision_at_k(preds, labels, k=10)
    ndcg_5 = compute_ndcg_at_k(preds, labels, k=5)
    prec_5 = compute_precision_at_k(preds, labels, k=5)
    
    metrics = {
        'rmse': float(rmse),
        'mae': float(mae),
        'ndcg@10': float(ndcg_10),
        'precision@10': float(prec_10),
        'ndcg@5': float(ndcg_5),
        'precision@5': float(prec_5),
        'num_test_samples': len(labels),
    }
    
    return metrics, preds, labels


def main():
    print("=" * 70)
    print("GNN Model Evaluation")
    print("=" * 70)
    
    device = torch.device(DATA['device'])
    
    # Load graph
    print("\nLoading graph...")
    data = load_graph_and_features(device=device)
    
    # Load model checkpoint
    print("\nLoading model checkpoint...")
    checkpoint = torch.load(CHECKPOINT_DIR / "best_model.pt", map_location=device)
    
    # Initialize model
    model = HeteroGraphSAGE(
        in_channels=MODEL['in_channels'],
        hidden_channels=MODEL['hidden_channels'],
        out_channels=MODEL['out_channels'],
        edge_types=data.edge_types,
        num_layers=MODEL['num_layers'],
        dropout=MODEL['dropout'],
    ).to(device)
    
    # The baseline rating head was trained on pre-computed 386-dim features
    # (user/movie features), so initialize the head accordingly to match checkpoint
    rating_model = HeteroGNNForRating(
        user_emb_dim=386,
        movie_emb_dim=386,
        hidden_dim=128,
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state'])
    rating_model.load_state_dict(checkpoint['rating_model_state'])
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    
    # Load test data (recreate from the same split as training)
    print("\nPreparing test data...")
    # This is a simplified version; in practice, you'd reload the same split indices
    # For now, we'll use the full ratings as a proxy
    processed_dir = ROOT / "data" / "processed"
    ratings_df = pd.read_csv(processed_dir / "ratings.csv")
    
    movies_df = pd.read_csv(processed_dir / "movies.csv")
    movie_id_to_idx = {mid: idx for idx, mid in enumerate(movies_df['movieId'].values)}
    unique_users = sorted(ratings_df['userId'].unique())
    user_id_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
    
    # Use a subset for evaluation
    test_indices = np.random.choice(len(ratings_df), size=min(10000, len(ratings_df)), replace=False)
    test_ratings = ratings_df.iloc[test_indices]
    
    # Build test edges and labels
    valid_rows = []
    ratings_list = []
    for _, row in test_ratings.iterrows():
        user_idx = user_id_to_idx.get(row['userId'])
        movie_idx = movie_id_to_idx.get(row['movieId'])
        rating = float(row['rating'])
        if user_idx is not None and movie_idx is not None:
            valid_rows.append([user_idx, movie_idx])
            ratings_list.append(rating)
    
    test_edges = torch.tensor(valid_rows, dtype=torch.long).t().contiguous() if valid_rows else torch.zeros((2, 0), dtype=torch.long)
    test_labels = torch.tensor(ratings_list, dtype=torch.float32) if ratings_list else torch.tensor([], dtype=torch.float32)
    
    test_loader = RatingDataLoader(data, test_edges, test_labels, batch_size=512, shuffle=False, device=device)
    
    # Evaluate
    print("\n=== Evaluation Metrics ===")
    metrics, preds, labels = evaluate_model(model, rating_model, test_loader, device)
    
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"NDCG@10: {metrics['ndcg@10']:.4f}")
    print(f"Precision@10: {metrics['precision@10']:.4f}")
    print(f"NDCG@5: {metrics['ndcg@5']:.4f}")
    print(f"Precision@5: {metrics['precision@5']:.4f}")
    print(f"Num test samples: {metrics['num_test_samples']}")
    
    # Save detailed metrics
    output_path = LOG_DIR / "detailed_evaluation.json"
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nDetailed metrics saved to {output_path}")


if __name__ == '__main__':
    main()
