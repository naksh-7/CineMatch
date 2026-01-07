"""Inspect checkpoint vs model state dict keys/shapes to help reconcile mismatches."""
import torch
from pathlib import Path
import pprint

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / 'checkpoints' / 'best_model.pt'

# ensure project root on path (like infer script)
import sys
sys.path.insert(0, str(ROOT))

print('Loading checkpoint:', CHECKPOINT)
ckpt = torch.load(CHECKPOINT, map_location='cpu')
model_state = ckpt.get('model_state', ckpt.get('state_dict', {}))
print('Checkpoint model_state keys/shapes (sample 50):')
for k, v in list(model_state.items())[:200]:
    try:
        print(k, tuple(v.shape))
    except Exception:
        print(k, '<unprintable>')

print('\nImporting local HeteroGraphSAGE (non-lazy)')
from scripts.models.hetero_gnn import HeteroGraphSAGE

model = HeteroGraphSAGE(in_channels=386, hidden_channels=128, out_channels=64, edge_types=[('person','acted_in','movie'),('person','directed_by','movie'),('user','rated','movie'),('user','tagged','movie')], num_layers=2, dropout=0.1, lazy_init=False)
print('\nLocal model keys/shapes:')
for k, v in model.state_dict().items():
    try:
        print(k, tuple(v.shape))
    except Exception:
        print(k, '<unprintable>')

ckpt_keys = set(model_state.keys())
model_keys = set(model.state_dict().keys())
print('\nKeys in checkpoint but not in model (sample 50):')
for k in list(ckpt_keys - model_keys)[:200]:
    print('  ', k)
print('\nKeys in model but not in checkpoint (sample 50):')
for k in list(model_keys - ckpt_keys)[:200]:
    print('  ', k)

# Print mismatched shapes for keys present in both
print('\nMismatched shapes for shared keys:')
for k in ckpt_keys & model_keys:
    ck_shape = tuple(model_state[k].shape) if hasattr(model_state[k], 'shape') else None
    mod_shape = tuple(model.state_dict()[k].shape) if hasattr(model.state_dict()[k], 'shape') else None
    if ck_shape != mod_shape:
        print(k, 'ckpt:', ck_shape, 'model:', mod_shape)

print('\nDone')