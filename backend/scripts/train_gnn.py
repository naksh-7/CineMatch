"""Training script for heterogeneous GNN on movie recommendation task.

Trains a HeteroGraphSAGE model to predict user-movie ratings using the
heterogeneous graph constructed in Phase 4.
"""

import os
os.environ['TORCH_CUDNN_ENABLED'] = '1'
os.environ['PYTHONWARNINGS'] = 'ignore'

import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
import logging
import traceback
import sys

# Ensure Python stdout is line-buffered so logs appear promptly
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# Configure basic logging to a dedicated error log file as well as stderr
ROOT = Path(__file__).resolve().parents[1]
LOG_FILE = ROOT / 'gnn_train_error.log'
logging.basicConfig(level=logging.INFO, handlers=[
    logging.FileHandler(LOG_FILE, mode='a'),
    logging.StreamHandler(sys.stderr)
])
log = logging.getLogger('train_gnn')


# Import local modules
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def setup_paths():
    """Setup directories and load config."""
    from config.gnn_config import EVALUATION, LOGGING
    ROOT = Path(__file__).resolve().parents[1]
    PROCESSED_DIR = ROOT / "data" / "processed"
    CHECKPOINT_DIR = ROOT / EVALUATION['checkpoint_dir']
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    LOG_DIR = ROOT / LOGGING['log_dir']
    LOG_DIR.mkdir(exist_ok=True)
    return ROOT, PROCESSED_DIR, CHECKPOINT_DIR, LOG_DIR


def create_id_mappings(processed_dir):
    """Create mappings from IDs to indices."""
    print("Creating ID mappings...")
    
    movies_df = pd.read_csv(processed_dir / "movies.csv")
    ratings_df = pd.read_csv(processed_dir / "ratings.csv")
    
    movie_id_to_idx = {mid: idx for idx, mid in enumerate(movies_df['movieId'].values)}
    unique_users = sorted(ratings_df['userId'].unique())
    user_id_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
    
    print(f"  Movies: {len(movie_id_to_idx)}")
    print(f"  Users: {len(user_id_to_idx)}")
    
    return movie_id_to_idx, user_id_to_idx, ratings_df


def train_epoch(model, rating_model, train_loader, optimizer, device, loss_config, training_config):
    """Train for one epoch."""
    model.train()
    rating_model.train()
    
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(train_loader, desc="Training", leave=False):
        user_features = batch['user_features'].to(device)
        movie_features = batch['movie_features'].to(device)
        labels = batch['labels'].to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        
        # For now, use pre-computed features directly
        # (Full GNN forward pass would require all nodes in memory)
        ratings = rating_model(user_features, movie_features)
        
        # Loss
        if loss_config['type'] == 'MSE':
            loss = nn.MSELoss()(ratings, labels)
        elif loss_config['type'] == 'L1':
            loss = nn.L1Loss()(ratings, labels)
        else:
            loss = nn.MSELoss()(ratings, labels)
        
        loss.backward()
        
        # Gradient clipping
        if training_config.get('gradient_clip', 0) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), training_config['gradient_clip'])
            torch.nn.utils.clip_grad_norm_(rating_model.parameters(), training_config['gradient_clip'])
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def train_epoch_gnn(model, rating_model, gnn_train_loader, optimizer, device, loss_config, training_config):
    """Train for one epoch using LinkNeighborLoader mini-batches.
    """
    model.train()
    rating_model.train()

    total_loss = 0.0
    num_batches = 0

    edge_type = ('user', 'rated', 'movie')
    for batch in tqdm(gnn_train_loader, desc="GNN Training", leave=False):
        # batch is a HeteroData with sampled node features and edges
        batch = batch.to(device)

        optimizer.zero_grad()

        # Forward pass: get node embeddings for the sampled subgraph
        x_dict = {k: v for k, v in batch.x_dict.items()}
        edge_index_dict = {k: v for k, v in batch.edge_index_dict.items()} if hasattr(batch, 'edge_index_dict') else batch.edge_index_dict
        embeddings = model(x_dict, edge_index_dict)

        # Extract edge_label_index and labels for the target relation
        # Try multiple possible attribute locations for compatibility
        if hasattr(batch[edge_type], 'edge_label_index'):
            edge_label_index = batch[edge_type].edge_label_index
        elif hasattr(batch, 'edge_label_index_dict') and edge_type in batch.edge_label_index_dict:
            edge_label_index = batch.edge_label_index_dict[edge_type]
        else:
            raise RuntimeError('edge_label_index not found in batch')

        if hasattr(batch[edge_type], 'edge_label'):
            labels = batch[edge_type].edge_label
        elif hasattr(batch, 'edge_label'):
            labels = batch.edge_label
        else:
            # Fallback: if the loader sets 'edge_label' on the batch dict
            labels = batch[edge_type].edge_label

        user_indices = edge_label_index[0].long()
        movie_indices = edge_label_index[1].long()

        user_emb = embeddings['user'][user_indices]
        movie_emb = embeddings['movie'][movie_indices]

        # Predict
        ratings = rating_model(user_emb, movie_emb)

        # Loss
        if loss_config['type'] == 'MSE':
            loss = nn.MSELoss()(ratings, labels.unsqueeze(1))
        elif loss_config['type'] == 'L1':
            loss = nn.L1Loss()(ratings, labels.unsqueeze(1))
        else:
            loss = nn.MSELoss()(ratings, labels.unsqueeze(1))

        loss.backward()

        # Gradient clipping
        if training_config.get('gradient_clip', 0) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), training_config['gradient_clip'])
            torch.nn.utils.clip_grad_norm_(rating_model.parameters(), training_config['gradient_clip'])

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


@torch.no_grad()
def evaluate_gnn(model, rating_model, gnn_loader, device):
    model.eval()
    rating_model.eval()
    all_preds = []
    all_labels = []

    edge_type = ('user', 'rated', 'movie')
    for batch in tqdm(gnn_loader, desc='GNN Evaluating', leave=False):
        batch = batch.to(device)

        x_dict = {k: v for k, v in batch.x_dict.items()}
        edge_index_dict = {k: v for k, v in batch.edge_index_dict.items()} if hasattr(batch, 'edge_index_dict') else batch.edge_index_dict
        embeddings = model(x_dict, edge_index_dict)

        if hasattr(batch[edge_type], 'edge_label_index'):
            edge_label_index = batch[edge_type].edge_label_index
        elif hasattr(batch, 'edge_label_index_dict') and edge_type in batch.edge_label_index_dict:
            edge_label_index = batch.edge_label_index_dict[edge_type]
        else:
            raise RuntimeError('edge_label_index not found in batch')

        labels = batch[edge_type].edge_label
        user_indices = edge_label_index[0].long()
        movie_indices = edge_label_index[1].long()
        user_emb = embeddings['user'][user_indices]
        movie_emb = embeddings['movie'][movie_indices]

        ratings = rating_model(user_emb, movie_emb).squeeze()
        all_preds.append(ratings.cpu())
        all_labels.append(labels.cpu())

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    rmse = torch.sqrt(torch.mean((preds - labels) ** 2)).item()
    mae = torch.mean(torch.abs(preds - labels)).item()
    return rmse, mae


@torch.no_grad()
def evaluate(model, rating_model, val_loader, device):
    """Evaluate model on validation set."""
    model.eval()
    rating_model.eval()
    
    all_preds = []
    all_labels = []
    
    for batch in tqdm(val_loader, desc="Evaluating", leave=False):
        user_features = batch['user_features'].to(device)
        movie_features = batch['movie_features'].to(device)
        labels = batch['labels'].to(device)
        
        user_pred = user_features
        movie_pred = movie_features
        ratings = rating_model(user_pred, movie_pred).squeeze()
        
        all_preds.append(ratings.cpu())
        all_labels.append(labels.cpu())
    
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    
    # Compute metrics
    rmse = torch.sqrt(torch.mean((preds - labels) ** 2)).item()
    mae = torch.mean(torch.abs(preds - labels)).item()
    
    return rmse, mae


def main():
    """Main training pipeline."""
    # Lazy import to avoid long PyG initialization
    from scripts.models.hetero_gnn import HeteroGraphSAGE, HeteroGNNForRating
    from scripts.data_utils import load_graph_and_features, create_train_val_test_split, prepare_rating_edges, RatingDataLoader, prepare_link_neighbor_loaders
    from config.gnn_config import MODEL, TRAINING, DATA, LOSS, EVALUATION, LOGGING
    
    # Setup paths
    ROOT, PROCESSED_DIR, CHECKPOINT_DIR, LOG_DIR = setup_paths()
    
    print("=" * 70)
    print("Phase 5: GNN Training for Movie Recommendation")
    print("=" * 70)
    
    device = torch.device(DATA['device'])
    print(f"\nUsing device: {device}")
    
    # Load graph and features
    print("\n=== Loading Data ===")
    data = load_graph_and_features(device=device)
    
    # Create ID mappings and load ratings
    movie_id_to_idx, user_id_to_idx, ratings_df = create_id_mappings(PROCESSED_DIR)
    
    # Train/val/test split
    print("\nSplitting data...")
    split_indices = create_train_val_test_split(
        ratings_df,
        train_ratio=DATA['train_ratio'],
        val_ratio=DATA['val_ratio'],
        test_ratio=DATA['test_ratio'],
        random_seed=DATA['random_seed'],
    )
    train_idx, val_idx, test_idx = split_indices
    print(f"  Train: {len(train_idx)}")
    print(f"  Val: {len(val_idx)}")
    print(f"  Test: {len(test_idx)}")
    
    # Prepare rating edges
    print("\nPreparing rating edges...")
    rating_splits = prepare_rating_edges(ratings_df, movie_id_to_idx, user_id_to_idx, split_indices)
    print(f"  Train edges: {rating_splits['train']['edges'].shape[1]}")
    print(f"  Val edges: {rating_splits['val']['edges'].shape[1]}")
    print(f"  Test edges: {rating_splits['test']['edges'].shape[1]}")
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader = RatingDataLoader(
        data, rating_splits['train']['edges'], rating_splits['train']['labels'],
        batch_size=TRAINING['batch_size'], shuffle=True, device=device
    )
    val_loader = RatingDataLoader(
        data, rating_splits['val']['edges'], rating_splits['val']['labels'],
        batch_size=TRAINING['batch_size'], shuffle=False, device=device
    )
    test_loader = RatingDataLoader(
        data, rating_splits['test']['edges'], rating_splits['test']['labels'],
        batch_size=TRAINING['batch_size'], shuffle=False, device=device
    )
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    # Optionally create LinkNeighborLoader-based loaders for mini-batch GNN training
    if TRAINING.get('use_link_neighbor_loader', False):
        print("\nCreating LinkNeighborLoader-based loaders for GNN mini-batch training...")
        gnn_train_loader, gnn_val_loader, gnn_test_loader = prepare_link_neighbor_loaders(
            data,
            train_edges=rating_splits['train']['edges'],
            train_labels=rating_splits['train']['labels'],
            val_edges=rating_splits['val']['edges'],
            val_labels=rating_splits['val']['labels'],
            test_edges=rating_splits['test']['edges'],
            test_labels=rating_splits['test']['labels'],
            batch_size=TRAINING['batch_size'],
            num_neighbors=[10, 10],
            shuffle=True,
        )
        # Print approximate loader sizes
        print(f"  GNN Train batches (approx): {len(gnn_train_loader)}")
        print(f"  GNN Val batches (approx): {len(gnn_val_loader)}")
        print(f"  GNN Test batches (approx): {len(gnn_test_loader)}")

    # If in dry-run mode, exit early after loader creation
    if os.environ.get('DRY_RUN', '0') == '1':
        print("Dry-run mode: exiting after loader creation")
        return
    
    # Initialize model
    print("\n=== Initializing Model ===")
    model = HeteroGraphSAGE(
        in_channels=MODEL['in_channels'],
        hidden_channels=MODEL['hidden_channels'],
        out_channels=MODEL['out_channels'],
        edge_types=data.edge_types,
        num_layers=MODEL['num_layers'],
        dropout=MODEL['dropout'],
    ).to(device)
    
    rating_model = HeteroGNNForRating(
        user_emb_dim=386,
        movie_emb_dim=386,
        hidden_dim=128,
    ).to(device)
    
    print(f"  Model: {MODEL['type']}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"  Rating head parameters: {sum(p.numel() for p in rating_model.parameters() if p.requires_grad)}")
    
    # Optimizer
    all_params = list(model.parameters()) + list(rating_model.parameters())
    if TRAINING['optimizer'] == 'Adam':
        optimizer = optim.Adam(all_params, lr=TRAINING['learning_rate'], weight_decay=TRAINING['weight_decay'])
    else:
        optimizer = optim.SGD(all_params, lr=TRAINING['learning_rate'], weight_decay=TRAINING['weight_decay'])
    
    # Training loop
    print("\n=== Training ===")
    best_val_rmse = float('inf')
    patience_counter = 0
    metrics_log = []
    
    start_time = time.time()
    
    for epoch in range(TRAINING['epochs']):
        epoch_start = time.time()
        
        # Train
        if TRAINING.get('use_link_neighbor_loader', False):
            train_loss = train_epoch_gnn(model, rating_model, gnn_train_loader, optimizer, device, LOSS, TRAINING)
        else:
            train_loss = train_epoch(model, rating_model, train_loader, optimizer, device, LOSS, TRAINING)
        
        # Evaluate
        if (epoch + 1) % EVALUATION['eval_every'] == 0:
            if TRAINING.get('use_link_neighbor_loader', False):
                val_rmse, val_mae = evaluate_gnn(model, rating_model, gnn_val_loader, device)
                test_rmse, test_mae = evaluate_gnn(model, rating_model, gnn_test_loader, device)
            else:
                val_rmse, val_mae = evaluate(model, rating_model, val_loader, device)
                test_rmse, test_mae = evaluate(model, rating_model, test_loader, device)
            
            epoch_time = time.time() - epoch_start
            
            print(f"Epoch {epoch+1:3d} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val RMSE: {val_rmse:.4f} | "
                  f"Test RMSE: {test_rmse:.4f} | "
                  f"Time: {epoch_time:.2f}s")
            
            metrics_log.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_rmse': val_rmse,
                'val_mae': val_mae,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
            })
            
            # Early stopping
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                patience_counter = 0
                
                # Save best model
                if EVALUATION['save_best']:
                    torch.save({
                        'model_state': model.state_dict(),
                        'rating_model_state': rating_model.state_dict(),
                        'epoch': epoch + 1,
                        'val_rmse': val_rmse,
                    }, CHECKPOINT_DIR / "best_model.pt")
                    print(f"  -> Saved best model (Val RMSE: {val_rmse:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= TRAINING['early_stopping_patience']:
                    print(f"Early stopping at epoch {epoch+1} (patience {patience_counter})")
                    break
    
    total_time = time.time() - start_time
    
    # Final evaluation on test set
    print("\n=== Final Evaluation ===")
    checkpoint = torch.load(CHECKPOINT_DIR / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    rating_model.load_state_dict(checkpoint['rating_model_state'])
    
    test_rmse, test_mae = evaluate(model, rating_model, test_loader, device)
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    
    # Save results
    results = {
        'model_config': MODEL,
        'training_config': TRAINING,
        'best_epoch': checkpoint['epoch'],
        'best_val_rmse': float(checkpoint['val_rmse']),
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'total_training_time_seconds': total_time,
        'metrics_log': metrics_log,
    }
    
    with open(LOG_DIR / "baseline_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Results saved to {LOG_DIR / 'baseline_metrics.json'}")
    print(f"Best model saved to {CHECKPOINT_DIR / 'best_model.pt'}")


if __name__ == '__main__':
    try:
        # Write immediate startup heartbeat to logs
        pid = os.getpid()
        start_msg = f"RUN STARTED pid={pid} time={time.strftime('%Y-%m-%d %H:%M:%S')}"
        print(start_msg, flush=True)
        log.info(start_msg)

        main()

        finish_msg = f"RUN FINISHED pid={pid} time={time.strftime('%Y-%m-%d %H:%M:%S')}"
        print(finish_msg, flush=True)
        log.info(finish_msg)
    except Exception as e:
        tb = traceback.format_exc()
        # Log to error file and stderr immediately
        print("UNCAUGHT EXCEPTION", flush=True)
        print(tb, flush=True)
        log.error("UNCAUGHT EXCEPTION:\n%s", tb)
        # Re-raise so Start-Process still notices non-zero exit if needed
        raise

