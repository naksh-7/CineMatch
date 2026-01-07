"""Heterogeneous GNN model definitions for movie recommendation.

Supports:
- GraphSAGE: Simple, scalable message-passing baseline
- RGCN: Relation-aware convolution for heterogeneous graphs
- HGT: Hierarchical graph transformer (advanced)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, RGCNConv, Linear
from torch_geometric.nn.dense import Linear as DenseLinear


class HeteroGraphSAGE(nn.Module):
    """Heterogeneous GraphSAGE with optional lazy initialization for checkpoint compatibility.

    When `lazy_init=True` the SAGEConv layers are created with shape hints of (-1,-1)
    which allows PyG to infer input dimensions on the first forward pass and avoid
    manual input-projection/shape handling. This is useful when your saved
    checkpoint's parameter shapes do not match the locally-specified `in_channels`.
    """

    def __init__(self, in_channels=None, hidden_channels=128, out_channels=64, edge_types=None, num_layers=2, dropout=0.1, lazy_init=False, add_rating_head=False):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.lazy_init = bool(lazy_init)

        # Whether to expose a rating head (some checkpoints include this)
        self.add_rating_head = bool(add_rating_head)
        if self.add_rating_head:
            self.rating_head = Linear(self.out_channels, 1)

        # In non-lazy mode we keep an input projection to align raw features -> hidden
        self.input_proj = None
        if not self.lazy_init and (self.in_channels is not None) and self.in_channels != self.hidden_channels:
            self.input_proj = nn.Linear(self.in_channels, self.hidden_channels)

        if edge_types is None:
            edge_types = []

        # Build heterogeneous convolutions --- support explicit per-layer per-edge dims via `edge_dims`
        # `edge_dims` (if provided) should be a list of dicts with keys (src,rel,dst) -> (in_dim, out_dim)
        self.convs = nn.ModuleList()
        if hasattr(self, 'edge_dims') and self.edge_dims is not None:
            # use explicit edge_dims specification
            for layer_idx in range(len(self.edge_dims)):
                conv_dict = {}
                layer_map = self.edge_dims[layer_idx]
                for (src, rel, dst), (in_dim, out_dim) in layer_map.items():
                    conv_dict[(src, rel, dst)] = SAGEConv((in_dim, in_dim), out_dim, flow='source_to_target')
                self.convs.append(HeteroConv(conv_dict, aggr='mean'))
        else:
            for layer_idx in range(self.num_layers):
                out_dim = self.hidden_channels if layer_idx < self.num_layers - 1 else self.out_channels

                conv_dict = {}
                for src, rel, dst in edge_types:
                    if self.lazy_init:
                        # Let PyG infer input dims on first forward pass (-1 indicates unknown)
                        conv_dict[(src, rel, dst)] = SAGEConv((-1, -1), out_dim)
                    else:
                        in_dim = self.in_channels if layer_idx == 0 else self.hidden_channels
                        conv_dict[(src, rel, dst)] = SAGEConv((in_dim, in_dim), out_dim, flow='source_to_target')

                self.convs.append(HeteroConv(conv_dict, aggr='mean'))

    def forward(self, x_dict, edge_index_dict):
        # In non-lazy mode, apply input projection where appropriate
        if not self.lazy_init and self.input_proj is not None:
            for nt, val in list(x_dict.items()):
                if val is None:
                    continue
                if val.dim() == 2 and (self.in_channels is None or val.shape[1] == self.in_channels):
                    x_dict[nt] = self.input_proj(val)

        for layer_idx, conv in enumerate(self.convs):


            # HeteroConv will call each underlying SAGEConv for edge types that exist
            new_x = conv(x_dict, edge_index_dict)

            merged = {}
            for nt, old_val in x_dict.items():
                new_val = new_x.get(nt) if isinstance(new_x, dict) else None
                if new_val is None:
                    # preserve prior representation for node types that received no messages
                    merged[nt] = old_val
                else:
                    if layer_idx < self.num_layers - 1:
                        merged[nt] = F.relu(new_val)
                    else:
                        merged[nt] = new_val

            if layer_idx < self.num_layers - 1:
                merged = {k: self.dropout(v) for k, v in merged.items()}

            # If we have explicit edge_dims, ensure node features match expected input dims for next layer
            next_layer_idx = layer_idx + 1
            if hasattr(self, 'edge_dims') and self.edge_dims is not None and next_layer_idx < len(self.edge_dims):
                next_map = self.edge_dims[next_layer_idx]
                # create ModuleDict container if needed
                if not hasattr(self, 'layer_projs'):
                    self.layer_projs = nn.ModuleDict()
                layer_key = str(next_layer_idx)
                if layer_key not in self.layer_projs:
                    self.layer_projs[layer_key] = nn.ModuleDict()

                for nt in list(merged.keys()):
                    # find expected in dim for this node type if it appears as source in next layer
                    expected_in = None
                    for (src, rel, dst), (in_dim, out_dim) in next_map.items():
                        if src == nt:
                            expected_in = in_dim
                            break
                    if expected_in is None:
                        continue
                    actual_dim = merged[nt].shape[1]
                    if actual_dim != expected_in:
                        # create a projection module lazily if not exists
                        if nt not in self.layer_projs[layer_key]:
                            proj = nn.Linear(actual_dim, expected_in)
                            # move to same device as one of the merged tensors
                            proj.to(merged[nt].device)
                            self.layer_projs[layer_key][nt] = proj
                        else:
                            proj = self.layer_projs[layer_key][nt]
                        merged[nt] = proj(merged[nt])

            x_dict = merged

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
