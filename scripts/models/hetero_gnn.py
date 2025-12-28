"""Heterogeneous GNN model definitions for movie recommendation.

Supports:
- GraphSAGE: Simple, scalable message-passing baseline
- RGCN: Relation-aware convolution for heterogeneous graphs
- HGT: Hierarchical graph transformer (advanced)
"""

import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, SAGEConv, RGCNConv, Linear
from torch_geometric.nn.dense import Linear as DenseLinear


class HeteroGraphSAGE(nn.Module):
    """Heterogeneous GraphSAGE for multi-relation graphs.
    
    Message-passing on all edge types, aggregated with mean pooling.
    Outputs embeddings for all node types.
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels, edge_types, num_layers=2, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        
        # Build heterogeneous convolutions for each layer
        self.convs = nn.ModuleList()
        for layer_idx in range(num_layers):
            # Input channels for this layer
            input_dim = in_channels if layer_idx == 0 else hidden_channels
            output_dim = hidden_channels if layer_idx < num_layers - 1 else out_channels
            
            conv_dict = {}
            for src, rel, dst in edge_types:
                conv_key = (src, rel, dst)
                conv_dict[conv_key] = SAGEConv((input_dim, input_dim), output_dim, flow='source_to_target')
            
            self.convs.append(HeteroConv(conv_dict, aggr='mean'))
        
        # Prediction head for rating prediction (optional)
        self.rating_head = Linear(out_channels, 1)
    
    def forward(self, x_dict, edge_index_dict):
        """Forward pass through heterogeneous graph.
        
        Args:
            x_dict: {node_type: feature_tensor} for all node types
            edge_index_dict: {(src, rel, dst): edge_index} for all edge types
        
        Returns:
            Embedding dict: {node_type: embedding_tensor}
        """
        for layer_idx, conv in enumerate(self.convs):
            # Message passing
            x_dict = conv(x_dict, edge_index_dict)
            
            # Apply activation (except last layer for embeddings)
            if layer_idx < self.num_layers - 1:
                x_dict = {k: v.relu() for k, v in x_dict.items()}
                x_dict = {k: self.dropout(v) for k, v in x_dict.items()}
        
        return x_dict
    
    def predict_ratings(self, embeddings):
        """Predict ratings from movie embeddings.
        
        Args:
            embeddings: Movie node embeddings (batch_size, out_channels)
        
        Returns:
            Ratings (batch_size, 1)
        """
        return self.rating_head(embeddings)


class HeteroRGCN(nn.Module):
    """Relational Graph Convolutional Network for heterogeneous graphs.
    
    Uses relation-specific transformations for more expressive message passing.
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations, num_layers=2, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        
        self.convs = nn.ModuleList()
        for layer_idx in range(num_layers):
            input_dim = in_channels if layer_idx == 0 else hidden_channels
            output_dim = hidden_channels if layer_idx < num_layers - 1 else out_channels
            
            self.convs.append(
                RGCNConv(input_dim, output_dim, num_relations=num_relations, aggr='mean')
            )
        
        self.rating_head = Linear(out_channels, 1)
    
    def forward(self, x, edge_index, edge_type):
        """Forward pass.
        
        Args:
            x: Node features (num_nodes, in_channels)
            edge_index: Edge connectivity (2, num_edges)
            edge_type: Relation type per edge (num_edges,)
        
        Returns:
            Node embeddings (num_nodes, out_channels)
        """
        for layer_idx, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)
            
            if layer_idx < self.num_layers - 1:
                x = x.relu()
                x = self.dropout(x)
        
        return x
    
    def predict_ratings(self, embeddings):
        """Predict ratings from embeddings."""
        return self.rating_head(embeddings)


class HeteroGNNForRating(nn.Module):
    """Wrapper for rating prediction task.
    
    Takes user and movie embeddings, computes interaction, and predicts rating.
    """
    
    def __init__(self, user_emb_dim=386, movie_emb_dim=386, hidden_dim=128):
        super().__init__()
        self.user_emb_dim = user_emb_dim
        self.movie_emb_dim = movie_emb_dim
        
        # Interaction layer: concatenate user + movie embeddings
        interaction_dim = user_emb_dim + movie_emb_dim
        
        self.mlp = nn.Sequential(
            Linear(interaction_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, user_emb, movie_emb):
        """Predict rating from user and movie embeddings.
        
        Args:
            user_emb: User embeddings (batch_size, user_emb_dim)
            movie_emb: Movie embeddings (batch_size, movie_emb_dim)
        
        Returns:
            Predicted ratings (batch_size, 1)
        """
        interaction = torch.cat([user_emb, movie_emb], dim=1)
        return self.mlp(interaction)
