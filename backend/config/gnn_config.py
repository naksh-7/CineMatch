"""Hyperparameter configuration for GNN training."""

# Model configuration
MODEL = {
    'type': 'HeteroGraphSAGE',  # Options: HeteroGraphSAGE, HeteroRGCN, HGT
    'in_channels': 386,  # Input feature dimension (from movie embeddings)
    'hidden_channels': 128,  # Hidden layer dimension
    'out_channels': 64,  # Output embedding dimension
    'num_layers': 2,  # Number of message-passing layers
    'dropout': 0.1,  # Dropout rate
}

# Training configuration
TRAINING = {
    'batch_size': 512,
    'learning_rate': 0.001,
    'weight_decay': 0.0001,
    'epochs': 50,
    # Increased patience to allow the baseline head time to improve
    'early_stopping_patience': 20,
    'use_link_neighbor_loader': True,
    'optimizer': 'Adam',  # Options: Adam, SGD, AdamW
    'gradient_clip': 1.0,
    'warmup_epochs': 2,
}

# Data configuration
DATA = {
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    'random_seed': 42,
    'device': 'cuda',  # Options: cuda, cpu
    'pin_memory': True,
}

# Loss configuration
LOSS = {
    'type': 'MSE',  # Options: MSE (regression), BCE (classification), L1
    'reduction': 'mean',
}

# Evaluation configuration
EVALUATION = {
    'metrics': ['RMSE', 'MAE', 'NDCG@10', 'Precision@10'],
    'eval_every': 5,  # Evaluate every N epochs
    'save_best': True,
    'checkpoint_dir': 'checkpoints',
}

# Logging configuration
LOGGING = {
    'log_dir': 'logs',
    'log_every': 100,
    'save_metrics_every': 1,
    'use_tensorboard': False,
}
