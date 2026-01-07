"""Inference script: FAISS retrieval + optional GNN re-ranking.

Usage examples:
    # Content-based top-10 from seed titles
    python scripts/infer_recommend_gnn.py --seed_titles "The Matrix","Inception" --top_k 10

    # Keyword-weighted probe
    python scripts/infer_recommend_gnn.py --seed_titles "The Matrix" --keywords "Action:2.0,Thriller:1.5" --top_k 10

    # Use GNN re-ranking (may be slower, requires loading graph + model)
    python scripts/infer_recommend_gnn.py --seed_titles "The Matrix" --use_gnn --top_k 10
"""

import argparse
import json
import os
from pathlib import Path
import numpy as np
import sys
import re

ROOT = Path(__file__).resolve().parents[1]
# Ensure project root on path for local imports
sys.path.insert(0, str(ROOT))

try:
    import faiss
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

import torch
import torch.nn as nn
from sklearn.preprocessing import normalize
from torch_geometric.nn import SAGEConv, HeteroConv

# Local utilities
from backend.scripts.data_utils import load_graph_and_features

ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = ROOT / "data" / "features"
PROCESSED_DIR = ROOT / "data" / "processed"
CHECKPOINT = ROOT / "checkpoints" / "best_model.pt"
MOVIE_FEATURES = FEATURE_DIR / "movie_features.npy"
MOVIE_NAMES = FEATURE_DIR / "movie_feature_names.txt"
MOVIES_CSV = PROCESSED_DIR / "movies.csv"


def build_faiss_index(features, index_path=None, use_gpu=False):
    """Build or load a FAISS index (inner-product on L2-normalized vectors).

    Returns: index
    """
    if not _HAS_FAISS:
        raise RuntimeError("faiss is not available in this environment")

    d = features.shape[1]
    # We'll use IndexFlatIP on normalized vectors for cosine similarity
    index = faiss.IndexFlatIP(d)
    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    index.add(features.astype(np.float32))
    if index_path:
        faiss.write_index(index, str(index_path))
    return index


def load_or_build_index(features, index_path=None, use_gpu=False):
    if index_path and Path(index_path).exists():
        if not _HAS_FAISS:
            raise RuntimeError("faiss not available to load index")
        return faiss.read_index(str(index_path))
    return build_faiss_index(features, index_path=index_path, use_gpu=use_gpu)


def parse_keywords(kstr):
    """Parse keywords string like 'Action:2.0,Comedy:1.5' into dict."""
    if not kstr:
        return {}
    pairs = [p.strip() for p in kstr.split(",") if p.strip()]
    out = {}
    for p in pairs:
        if ":" in p:
            k, v = p.split(":", 1)
            try:
                out[k.strip()] = float(v)
            except ValueError:
                out[k.strip()] = 1.0
        else:
            out[p] = 1.0
    return out


def normalize_title(s):
    """Normalize a title for robust matching: remove trailing year, move trailing articles, and strip punctuation."""
    s = str(s or '').strip().lower()
    # remove trailing year like '(1999)'
    s = re.sub(r'\(\d{4}\)$', '', s).strip()
    # convert 'Last, The' -> 'The Last'
    if ',' in s:
        parts = [p.strip() for p in s.split(',')]
        if len(parts) == 2:
            s = f"{parts[1]} {parts[0]}"
    # remove punctuation, collapse whitespace
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def match_titles_to_ids(movies_df, titles, verbose=False):
    """Match user-provided titles to movieIds using normalized exact / fuzzy matching.

    Returns a list of movieIds. Prints mapping when verbose=True.
    """
    # build normalized canonical map once
    canon_map = {}
    norm_titles = []
    for idx, row in movies_df.iterrows():
        title = str(row['title'])
        canon = normalize_title(title)
        canon_map.setdefault(canon, []).append((row['movieId'], title, idx))
        norm_titles.append(canon)

    ids = []
    for q in titles:
        q_orig = q.strip()
        qcanon = normalize_title(q_orig)
        candidates = canon_map.get(qcanon)
        if candidates:
            # prefer exact case-insensitive match among candidates
            chosen_mid, chosen_title = None, None
            for mid, title, _ in candidates:
                if title.strip().lower() == q_orig.lower():
                    chosen_mid, chosen_title = mid, title
                    break
            if chosen_mid is None:
                chosen_mid, chosen_title, _ = candidates[0]
            ids.append(chosen_mid)
            if verbose:
                print(f"Matched seed '{q_orig}' -> '{chosen_title}' (movieId={chosen_mid})")
            continue

        # fallback: check if normalized titles contain the query canon
        found_idx = None
        for i, nt in enumerate(norm_titles):
            if qcanon in nt:
                found_idx = i
                break
        if found_idx is not None:
            row = movies_df.iloc[found_idx]
            ids.append(row['movieId'])
            if verbose:
                print(f"Fuzzy matched seed '{q_orig}' -> '{row['title']}' (movieId={row['movieId']})")
            continue

        # last chance: substring on raw title
        matches = movies_df[movies_df['title'].str.lower().str.contains(q_orig.lower(), na=False)]
        if not matches.empty:
            ids.append(matches.iloc[0]['movieId'])
            if verbose:
                print(f"Substring matched seed '{q_orig}' -> '{matches.iloc[0]['title']}' (movieId={matches.iloc[0]['movieId']})")
            continue

        if verbose:
            print(f"No match found for seed '{q_orig}'")

    return ids


def build_keyword_vector(movies_df, movie_features, keywords):
    """Create a vector representation of keywords by averaging features of movies matching genre keywords.

    Simple approach: look for keyword in `genres` column and average their features.
    Returns vector same dim as movie_features.
    """
    d = movie_features.shape[1]
    kw_vec = np.zeros(d, dtype=np.float32)
    total_weight = 0.0
    for k, w in keywords.items():
        mask = movies_df['genres'].str.contains(k, case=False, na=False)
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            continue
        vec = movie_features[idxs].mean(axis=0)
        kw_vec += w * vec
        total_weight += w
    if total_weight > 0:
        kw_vec /= total_weight
    return kw_vec


def crete_pseudo_user_embedding(seed_ids, movies_df, movie_features):
    # seed_ids are MovieLens movieId values; map to row indices
    id_to_idx = {mid: idx for idx, mid in enumerate(movies_df['movieId'].values)}
    idxs = [id_to_idx[mid] for mid in seed_ids if mid in id_to_idx]
    if len(idxs) == 0:
        return None
    return movie_features[idxs].mean(axis=0)


def construct_subgraph(data, movie_node_ids, allowed_node_types=None):
    """Construct a small induced HeteroData subgraph around movie_node_ids.
    Collects movie nodes + direct person/user/tag neighbors and the edges between them.

    `allowed_node_types` (iterable) can be used to limit which node types are included
    (e.g., ['movie','person','tag']) to keep subgraphs small and avoid heavy user neighborhoods.

    Returns: sub_data (HeteroData) and mapping dicts (original_idx -> new_idx per node type)
    """
    if allowed_node_types is None:
        allowed = set(data.node_types)
    else:
        allowed = set(allowed_node_types)
    from torch_geometric.data import HeteroData
    import torch

    sub = HeteroData()

    # Collect node indices for each allowed type
    node_idx = {nt: set() for nt in allowed}
    # Add movie nodes
    if 'movie' not in allowed:
        raise ValueError('movie must be included in allowed_node_types')
    node_idx['movie'].update(movie_node_ids)

    # For each relation, find neighbors (only consider relations whose src/dst are allowed)
    for rel in data.edge_types:
        src, rel_name, dst = rel
        if src not in allowed or dst not in allowed:
            continue
        eidx = data[rel].edge_index
        # find edges where dst in movie_node_ids
        if dst == 'movie':
            mask = np.isin(eidx[1].cpu().numpy(), list(movie_node_ids))
            src_nodes = eidx[0, mask].cpu().numpy()
            node_idx[src].update(src_nodes.tolist())
        # allow general neighbor inclusion (e.g., user rated movie)
        else:
            # collect any neighbors adjacent to movie nodes (both directions)
            mask_dst = np.isin(eidx[1].cpu().numpy(), list(movie_node_ids))
            mask_src = np.isin(eidx[0].cpu().numpy(), list(movie_node_ids))
            if mask_dst.any():
                node_idx[src].update(eidx[0, mask_dst].cpu().numpy().tolist())
            if mask_src.any():
                node_idx[dst].update(eidx[1, mask_src].cpu().numpy().tolist())

    # Convert to sorted lists and map to contiguous indices
    maps = {}
    for nt in allowed:
        maps[nt] = {old: new for new, old in enumerate(sorted(node_idx[nt]))}
        # create x if exists
        if hasattr(data[nt], 'x'):
            if len(node_idx[nt]) > 0:
                old_x = data[nt].x[list(sorted(node_idx[nt]))]
                sub[nt].x = old_x.clone()
            else:
                # create empty feature tensor with original dtype and width
                orig_x = data[nt].x
                sub[nt].x = orig_x.new_zeros((0, orig_x.shape[1]))
        else:
            sub[nt].num_nodes = len(maps[nt])

    # Reconstruct edges for subgraph
    for src, rel_name, dst in data.edge_types:
        if src not in allowed or dst not in allowed:
            continue
        eidx = data[(src, rel_name, dst)].edge_index
        # keep edges where both src/dst are in maps
        src_arr = eidx[0].cpu().numpy()
        dst_arr = eidx[1].cpu().numpy()
        keep_mask = np.array([ (s in maps[src]) and (d in maps[dst]) for s,d in zip(src_arr, dst_arr) ])
        if keep_mask.sum() == 0:
            continue
        src_keep = [maps[src][s] for s, k in zip(src_arr, keep_mask) if k]
        dst_keep = [maps[dst][d] for d, k in zip(dst_arr, keep_mask) if k]
        edge_index = torch.tensor([src_keep, dst_keep], dtype=torch.long)
        sub[(src, rel_name, dst)].edge_index = edge_index

    # Return the constructed subgraph and the mapping dicts (original_idx -> sub_idx)
    return sub, maps


def get_lean_subgraph(data, movie_node_ids, max_user_neighbors=500, max_person_neighbors=200, rng=None):
    """Create a compact induced HeteroData subgraph around movie_node_ids.

    - Caps the total number of 'user' and 'person' nodes included to avoid explosion.
    - Returns (sub, maps) like construct_subgraph.
    """
    import torch
    import numpy as np
    from torch_geometric.data import HeteroData

    if rng is None:
        rng = np.random.RandomState(42)

    allowed = set(data.node_types)
    node_idx = {nt: set() for nt in allowed}
    node_idx['movie'].update(movie_node_ids)

    # Collect users/persons connected to movies, but cap them
    # For each relation where dst == 'movie', collect source nodes
    for src, rel_name, dst in data.edge_types:
        if dst != 'movie':
            continue
        eidx = data[(src, rel_name, dst)].edge_index
        dst_arr = eidx[1].cpu().numpy()
        src_arr = eidx[0].cpu().numpy()
        mask = np.isin(dst_arr, list(movie_node_ids))
        src_nodes = np.unique(src_arr[mask])
        if src == 'user':
            if len(src_nodes) > max_user_neighbors:
                src_nodes = rng.choice(src_nodes, size=max_user_neighbors, replace=False)
        if src == 'person':
            if len(src_nodes) > max_person_neighbors:
                src_nodes = rng.choice(src_nodes, size=max_person_neighbors, replace=False)
        node_idx[src].update(src_nodes.tolist())

    # Also include person nodes' movie neighbors (to get person->movie edges)
    # and tags connected directly to movies
    for src, rel_name, dst in data.edge_types:
        if src == 'person' and dst == 'movie':
            continue
        if dst == 'tag' and src == 'movie':
            eidx = data[(src, rel_name, dst)].edge_index
            mask = np.isin(eidx[1].cpu().numpy(), list(movie_node_ids))
            tag_nodes = np.unique(eidx[0].cpu().numpy()[mask])
            node_idx['tag'].update(tag_nodes.tolist())

    # Build subgraph from collected nodes (similar to construct_subgraph)
    sub = HeteroData()
    maps = {}
    for nt in allowed:
        ids = sorted(node_idx[nt])
        maps[nt] = {old: new for new, old in enumerate(ids)}
        if hasattr(data[nt], 'x'):
            if len(ids) > 0:
                sub[nt].x = data[nt].x[ids].clone()
            else:
                orig_x = data[nt].x
                sub[nt].x = orig_x.new_zeros((0, orig_x.shape[1]))
        else:
            sub[nt].num_nodes = len(ids)

    # Reconstruct filtered edges
    import torch
    for src, rel_name, dst in data.edge_types:
        eidx = data[(src, rel_name, dst)].edge_index
        src_arr = eidx[0].cpu().numpy()
        dst_arr = eidx[1].cpu().numpy()
        keep_mask = np.array([ (s in maps[src]) and (d in maps[dst]) for s,d in zip(src_arr, dst_arr) ])
        if keep_mask.sum() == 0:
            continue
        src_keep = [maps[src][s] for s, k in zip(src_arr, keep_mask) if k]
        dst_keep = [maps[dst][d] for d, k in zip(dst_arr, keep_mask) if k]
        edge_index = torch.tensor([src_keep, dst_keep], dtype=torch.long)
        sub[(src, rel_name, dst)].edge_index = edge_index

    return sub, maps

def main(args):
    print("Loading movie features...")
    movie_features = np.load(MOVIE_FEATURES)
    # features expect shape (N, D)

    movies_df = None
    if Path(MOVIES_CSV).exists():
        import pandas as pd
        movies_df = pd.read_csv(MOVIES_CSV)
    else:
        raise RuntimeError("movies.csv not found in processed data")

    # Normalize features for cosine search
    norm_feats = normalize(movie_features, axis=1)

    if _HAS_FAISS:
        index = load_or_build_index(norm_feats, index_path=str(ROOT / 'movie_faiss.index'), use_gpu=False)
    else:
        # fallback: brute-force search using numpy
        index = None

    # Resolve seed ids
    seed_ids = []
    if args.seed_titles:
        titles = [s.strip() for s in args.seed_titles.split(',') if s.strip()]
        seed_matches = match_titles_to_ids(movies_df, titles, verbose=True)
        seed_ids += seed_matches
    if args.seed_ids:
        seed_ids += [int(s) for s in args.seed_ids.split(',') if s.strip()]
    seed_ids = list(set(seed_ids))
    # Ensure seed ids are plain Python ints for JSON serialization
    seed_ids = [int(s) for s in seed_ids]

    keywords = parse_keywords(args.keywords)

    # Build query vector
    q_vecs = []
    if seed_ids:
        pseudo = crete_pseudo_user_embedding(seed_ids, movies_df, movie_features)
        if pseudo is not None:
            q_vecs.append(pseudo)
    if keywords:
        kw_vec = build_keyword_vector(movies_df, movie_features, keywords)
        if kw_vec is not None:
            q_vecs.append(kw_vec)
    if len(q_vecs) == 0:
        raise RuntimeError('No seed IDs or keywords provided')

    q = np.mean(np.stack(q_vecs, axis=0), axis=0)
    q = q.astype(np.float32)
    q_norm = q / (np.linalg.norm(q) + 1e-12)

    # FAISS retrieval
    n_candidates = args.n_candidates
    print(f"Retrieving top {n_candidates} candidates via FAISS...")
    if index is not None:
        D, I = index.search(np.expand_dims(q_norm, axis=0).astype(np.float32), n_candidates)
        candidate_idxs = I[0].tolist()
    else:
        sims = norm_feats.dot(q_norm)
        candidate_idxs = np.argsort(-sims)[:n_candidates].tolist()

    # Map indices -> movieId
    movie_ids = movies_df['movieId'].values
    candidate_movie_ids = [int(movie_ids[i]) for i in candidate_idxs]

    # Map to internal indices for graph operations (used by GNN and heuristic re-ranker)
    movie_id_to_idx = {mid: idx for idx, mid in enumerate(movies_df['movieId'].values)}
    candidate_internal = [movie_id_to_idx[mid] for mid in candidate_movie_ids if mid in movie_id_to_idx]
    candidate_internal = list(set(candidate_internal))
    seed_internal = [movie_id_to_idx[mid] for mid in seed_ids if mid in movie_id_to_idx]
    seed_internal = list(set(seed_internal))
    all_internal = list(set(candidate_internal) | set(seed_internal))

    # Precompute content similarity scores (cosine) for mix/fallback
    norm_feats_f32 = norm_feats.astype(np.float32)
    qn = q_norm.astype(np.float32)
    sims = norm_feats_f32.dot(qn)

    # final_scores will be set either by GNN re-ranking or fallback to content-based
    final_scores = None

    # Optionally re-rank with GNN
    if args.use_gnn:
        print('Loading graph and model for GNN re-ranking...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data = load_graph_and_features(device=device)

        # Load model checkpoint
        try:
            ckpt = torch.load(CHECKPOINT, map_location=device)
        except Exception as e:
            print(f"Failed to load checkpoint {CHECKPOINT}: {e}; falling back to content similarity")
            ckpt = None

        # Diagnostic: inspect checkpoint expected conv weight shapes to decide behavior
        use_lazy = bool(args.lazy_init)
        inferred = {}
        add_rating_head = False
        if ckpt is not None and 'model_state' in ckpt:
            print("Checkpoint conv weight shapes (lin_l / lin_r if present):")
            lin_shapes = {}
            edge_dims = {}
            for k, v in ckpt['model_state'].items():
                if 'convs.' in k and ('lin_l.weight' in k or 'lin_r.weight' in k):
                    try:
                        print(f"  {k}: {tuple(v.shape)}")
                        # parse the key into layer and edge triple: convs.<L>.convs.<person___acted_in___movie>.lin_l.weight
                        parts = k.split('.')
                        layer_part = parts[1] if len(parts) > 1 else None
                        if layer_part is not None and layer_part.isdigit():
                            layer_idx = int(layer_part)
                            lin_shapes.setdefault(layer_idx, []).append(tuple(v.shape))

                            # extract the <...> token containing edge triple
                            for p in parts:
                                if p.startswith('<') and p.endswith('>'):
                                    tri = p[1:-1]
                                    tri_parts = tri.split('___')
                                    if len(tri_parts) == 3:
                                        src, rel, dst = tri_parts
                                        # weight shape is (out, in)
                                        out_dim, in_dim = tuple(v.shape)
                                        edge_dims.setdefault(layer_idx, {})[(src, rel, dst)] = (in_dim, out_dim)
                    except Exception:
                        print(f"  {k}: <unprintable shape>")

            # derive dims from first available conv shapes and prepare model edge_dims
            if lin_shapes:
                # layer 0
                first_layer = lin_shapes.get(0)
                if first_layer and len(first_layer) > 0:
                    out0, in0 = first_layer[0]
                    inferred['in_channels'] = in0
                    inferred['hidden_channels'] = out0
                # layer 1
                second_layer = lin_shapes.get(1)
                if second_layer and len(second_layer) > 0:
                    out1, in1 = second_layer[0]
                    inferred['out_channels'] = out1

            if edge_dims:
                # convert edge_dims dict to list ordered by layer idx
                max_layer = max(edge_dims.keys())
                inferred_edge_dims = [edge_dims.get(i, {}) for i in range(max_layer + 1)]
                print(f"Extracted explicit edge dims for {len(inferred_edge_dims)} layers")
            else:
                inferred_edge_dims = None

            # detect rating head
            if 'rating_head.weight' in ckpt['model_state']:
                add_rating_head = True

            if inferred:
                print(f"Inferred dims from checkpoint: {inferred}")
                # if user explicitly requested lazy_init, keep it; otherwise prefer concrete dims
                if not args.lazy_init:
                    use_lazy = False
                else:
                    use_lazy = True

        from backend.scripts.models.hetero_gnn import HeteroGraphSAGE
        model = HeteroGraphSAGE(
            in_channels=(int(inferred['in_channels']) if 'in_channels' in inferred else (int(args.in_channels) if not args.lazy_init else None)),
            hidden_channels=(int(inferred['hidden_channels']) if 'hidden_channels' in inferred else int(args.hidden_channels)),
            out_channels=(int(inferred['out_channels']) if 'out_channels' in inferred else int(args.out_channels)),
            edge_types=data.edge_types,
            num_layers=(len(inferred_edge_dims) if inferred_edge_dims is not None else int(args.num_layers)),
            dropout=float(args.dropout),
            lazy_init=use_lazy,
            add_rating_head=add_rating_head,
        ).to(device)

        # If we extracted explicit per-edge dims from the checkpoint, attach them to the model
        if inferred_edge_dims is not None:
            model.edge_dims = inferred_edge_dims
            # Avoid pre-projection when we are matching checkpoint-shaped convs
            model.input_proj = None

            # rebuild convs to use explicit dims
            model.convs = nn.ModuleList()
            for layer_idx, layer_map in enumerate(model.edge_dims):
                conv_dict = {}
                for (src, rel, dst), (in_dim, out_dim) in layer_map.items():
                    conv_dict[(src, rel, dst)] = SAGEConv((in_dim, in_dim), out_dim, flow='source_to_target')
                model.convs.append(HeteroConv(conv_dict, aggr='mean'))

            # move newly created conv modules to the device of the model
            for conv in model.convs:
                conv.to(device)
            if hasattr(model, 'rating_head'):
                model.rating_head.to(device)

        if inferred:
            print(f"Built model with in={model.in_channels}, hidden={model.hidden_channels}, out={model.out_channels}, lazy={model.lazy_init}, rating_head={hasattr(model, 'rating_head')}")
        if ckpt is not None and 'model_state' in ckpt:
            # allow missing new projection weights (fallback_proj) when loading older checkpoints
            try:
                model.load_state_dict(ckpt['model_state'])
            except RuntimeError as e:
                print(f"State dict mismatch (expected for updated local model). Loading with strict=False: {e}")
                model.load_state_dict(ckpt['model_state'], strict=False)
        model.eval()

        # Debug: print conv parameter shapes to ensure compatibility with checkpoint
        try:
            for i, conv in enumerate(model.convs):
                print(f"Layer {i} convs:")
                for rel_key, sage in conv.convs.items():
                    lin_l = getattr(sage, 'lin_l', None)
                    lin_r = getattr(sage, 'lin_r', None)
                    w1 = None if lin_l is None else tuple(lin_l.weight.shape)
                    w2 = None if lin_r is None else tuple(lin_r.weight.shape)
                    print(f"  {rel_key}: lin_l={w1}, lin_r={w2}")
        except Exception as _:
            pass

        # Map movie IDs to internal movie indices
        movie_id_to_idx = {mid: idx for idx, mid in enumerate(movies_df['movieId'].values)}
        candidate_internal = [movie_id_to_idx[mid] for mid in candidate_movie_ids if mid in movie_id_to_idx]
        candidate_internal = list(set(candidate_internal))
        # include seed movies to compute seed embeddings in GNN space
        seed_internal = [movie_id_to_idx[mid] for mid in seed_ids if mid in movie_id_to_idx]
        seed_internal = list(set(seed_internal))

        all_internal = list(set(candidate_internal) | set(seed_internal))

        # Build a lean subgraph (cap user/person neighbors) around seeds + candidates
        sub, maps = get_lean_subgraph(data, all_internal, max_user_neighbors=args.max_user_neighbors, max_person_neighbors=args.max_person_neighbors)

        # Ensure node features exist and pad/truncate to desired in_channels safely
        desired_in = int(args.in_channels)
        padded_nodes = []
        for nt in sub.node_types:
            num_nodes = getattr(sub[nt], 'num_nodes', None)
            x = getattr(sub[nt], 'x', None)
            if num_nodes is None and x is None:
                continue
            if x is None:
                # create zero features
                sub[nt].x = torch.zeros((int(num_nodes), desired_in), dtype=torch.float32)
                padded_nodes.append((nt, 0, desired_in))
            else:
                if x.shape[1] != desired_in:
                    new = torch.zeros((x.shape[0], desired_in), dtype=torch.float32)
                    new[:, :x.shape[1]] = x.float()
                    sub[nt].x = new
                    padded_nodes.append((nt, x.shape[1], desired_in))
        if padded_nodes:
            print(f"Padded/created node features to match in_channels={desired_in}: {padded_nodes}")

        used_gnn = False
        with torch.no_grad():
            try:
                # Debug: print subgraph node feature summary before moving to device
                print("Subgraph node summary before to(device):")
                for nt in sub.node_types:
                    x = getattr(sub[nt], 'x', None)
                    num = getattr(sub[nt], 'num_nodes', None)
                    if x is None:
                        print(f"  {nt}: x=None, num_nodes={num}")
                    else:
                        print(f"  {nt}: x.shape={tuple(x.shape)}, dtype={getattr(x,'dtype',None)}")

                sub = sub.to(device)

                # Debug: print subgraph node feature summary after moving to device
                print("Subgraph node summary after to(device):")
                for nt in sub.node_types:
                    x = getattr(sub[nt], 'x', None)
                    if x is None:
                        print(f"  {nt}: x=None")
                    else:
                        print(f"  {nt}: x.shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device}")

                # Build x_dict_in: only include node types that have >=1 node to avoid empty-type aggregation issues
                x_dict_in = {}
                for nt in sub.node_types:
                    x = getattr(sub[nt], 'x', None)
                    if x is not None and x.shape[0] > 0:
                        x_dict_in[nt] = x.to(device)

                # Build edge_index_dict_in only for relations whose src/dst are present in x_dict_in and edge non-empty
                edge_index_dict_in = {}
                for rel in data.edge_types:
                    src, rel_name, dst = rel
                    if rel in sub.edge_index_dict and src in x_dict_in and dst in x_dict_in:
                        e = sub.edge_index_dict[rel]
                        if getattr(e, 'numel', lambda: 0)() > 0:
                            # ensure edge_index on same device
                            edge_index_dict_in[rel] = e.to(device)

                print(f"Passing to model: x types={list(x_dict_in.keys())}, relations={list(edge_index_dict_in.keys())}")

                embeddings = model(x_dict_in, edge_index_dict_in)
            except Exception as e:
                print(f"Exception during model forward: {e}")
                # print more debugging info
                try:
                    print(f"sub.x_dict keys: {list(sub.x_dict.keys())}")
                    for k, v in sub.x_dict.items():
                        print(f"  {k}: type={type(v)}, val={None if v is None else getattr(v, 'shape', str(v))}")
                except Exception as e2:
                    print(f"Failed to inspect sub.x_dict: {e2}")
                raise

            # Process embeddings and compute GNN scores
            try:
                if not isinstance(embeddings, dict) or 'movie' not in embeddings:
                    print("Model did not return 'movie' embeddings; skipping GNN re-ranking")
                else:
                    movie_embs = embeddings['movie']  # [num_sub_movies, dim]
                    if movie_embs is None:
                        raise RuntimeError('movie embeddings are None')

                    print(f"movie_embs.shape={tuple(movie_embs.shape)}")

                    # reverse mapping: sub idx -> original idx
                    sub_to_old = {v: k for k, v in maps['movie'].items()}

                    # collect sub indices for seeds and candidates
                    seed_sub_idxs = [maps['movie'][orig] for orig in seed_internal if orig in maps['movie']]
                    cand_sub_idxs = [maps['movie'][orig] for orig in candidate_internal if orig in maps['movie']]

                    print(f"seed_sub_idxs n={len(seed_sub_idxs)}, cand_sub_idxs n={len(cand_sub_idxs)}")

                    if len(seed_sub_idxs) == 0 or len(cand_sub_idxs) == 0:
                        print('No seeds or candidates present in subgraph after mapping; skipping GNN re-ranking')
                    else:
                        # compute mean seed embedding in GNN space
                        seed_emb = movie_embs[seed_sub_idxs].mean(dim=0, keepdim=True)
                        seed_emb = torch.nn.functional.normalize(seed_emb, dim=1)
                        cand_embs = movie_embs[cand_sub_idxs]
                        cand_embs = torch.nn.functional.normalize(cand_embs, dim=1)

                        # compute cosine similarities in GNN space
                        sims_gnn = torch.nn.functional.cosine_similarity(cand_embs, seed_emb.expand_as(cand_embs), dim=1).cpu().numpy()

                        # map back to movieIds and build score mapping
                        gnn_scores_by_movieId = {}
                        for sub_idx, score in zip(cand_sub_idxs, sims_gnn):
                            orig_idx = sub_to_old[sub_idx]
                            mid = int(movie_ids[orig_idx])
                            gnn_scores_by_movieId[mid] = float(score)

                        # combine content and GNN scores using alpha
                        alpha = float(args.gnn_alpha) if hasattr(args, 'gnn_alpha') else 1.0
                        final_scores = []
                        for midx in candidate_idxs:
                            mid = int(movie_ids[midx])
                            content_score = float(sims[midx])
                            gnn_score = gnn_scores_by_movieId.get(mid)
                            if gnn_score is not None:
                                combined = alpha * gnn_score + (1.0 - alpha) * content_score
                            else:
                                combined = content_score
                            final_scores.append((mid, combined))

                        print(f"GNN re-ranked {len(gnn_scores_by_movieId)} candidates using alpha={alpha}")
                        used_gnn = True
            except Exception as e:
                print(f"Error during GNN re-ranking: {e}; falling back to content similarity")

        if not used_gnn:
            final_scores = None

    # If GNN didn't set final_scores, optionally run the heuristic graph re-ranker
    if final_scores is None:
        # Fallback: create score list from FAISS similarities if GNN didn't run/succeed
        final_scores = [(int(movie_ids[idx]), float(sims[idx])) for idx in candidate_idxs]

    # 2. HYBRID HEURISTIC RE-RANKER (Runs if graph_rerank is True, regardless of GNN)
    if args.graph_rerank:
        print('Running aggressive hybrid heuristic re-ranker (Director Boost + Title Junk Penalty) ...')
        # Ensure graph loaded (may not be loaded if --use_gnn wasn't set)
        if 'data' not in locals():
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            data = load_graph_and_features(device=device)

        # Build movie -> person sets and movie -> user sets
        def build_association_sets(data):
            movie_persons = {}
            movie_directors = {} # New: Separate tracking for directors
            movie_users = {}
            
            # Check for directed_by relation specifically
            for src, rel_name, dst in data.edge_types:
                eidx = data[(src, rel_name, dst)].edge_index
                if eidx.numel() == 0: continue
                src_arr = eidx[0].cpu().numpy()
                dst_arr = eidx[1].cpu().numpy()

                # Logic for Persons (Actors + Directors)
                if (src == 'person' and dst == 'movie') or (src == 'movie' and dst == 'person'):
                    m_arr, p_arr = (dst_arr, src_arr) if dst == 'movie' else (src_arr, dst_arr)
                    for m, p in zip(m_arr, p_arr):
                        movie_persons.setdefault(int(m), set()).add(int(p))
                        if rel_name == 'directed_by' or rel_name == 'rev_directed_by':
                            movie_directors.setdefault(int(m), set()).add(int(p))

                # Logic for Users
                if (src == 'user' and dst == 'movie') or (src == 'movie' and dst == 'user'):
                    m_arr, u_arr = (dst_arr, src_arr) if dst == 'movie' else (src_arr, dst_arr)
                    for m, u in zip(m_arr, u_arr):
                        movie_users.setdefault(int(m), set()).add(int(u))
            
            return movie_persons, movie_directors, movie_users

        movie_persons, movie_directors, movie_users = build_association_sets(data)

        # Seeds' aggregated neighbor sets
        seed_persons = set()
        seed_directors = set()
        seed_users = set()
        for m in seed_internal:
            seed_persons.update(movie_persons.get(m, set()))
            seed_directors.update(movie_directors.get(m, set()))
            seed_users.update(movie_users.get(m, set()))

        # Parameters
        person_boost = float(args.person_boost)
        user_boost = float(args.user_boost)
        no_overlap_penalty = float(args.no_overlap_penalty)
        hw = float(args.heuristic_weight)
        director_boost = 3.0 # Your new requirement
        
        # Prepare Title Junk detection
        id_to_title = {row['movieId']: str(row['title']).lower() for _, row in movies_df.iterrows()}
        seed_titles = [id_to_title.get(mid, "") for mid in seed_ids]

        new_hybrid_scores = []
        for mid, base_score in final_scores:
            m_idx = movie_id_to_idx.get(mid)
            if m_idx is None:
                new_hybrid_scores.append((mid, base_score))
                continue

            cand_persons = movie_persons.get(m_idx, set())
            cand_directors = movie_directors.get(m_idx, set())
            cand_users = movie_users.get(m_idx, set())

            # 1. Person Jaccard
            inter_p = len(seed_persons & cand_persons)
            union_p = len(seed_persons | cand_persons)
            j_p = (inter_p / union_p) if union_p > 0 else 0.0

            # 2. Director Boost (Style Reward)
            d_match = len(seed_directors & cand_directors)
            d_boost = (director_boost * d_match) if d_match > 0 else 0.0

            # 3. User Jaccard
            j_u = 0.0
            if args.user_rerank:
                inter_u = len(seed_users & cand_users)
                union_u = len(seed_users | cand_users)
                j_u = (inter_u / union_u) if union_u > 0 else 0.0

            # Calculate total boost
            total_boost = ((j_p * person_boost) + (j_u * user_boost) + d_boost) * hw

            # Apply Boosts or Overlap Penalties
            if (j_p > 0.0) or (j_u > 0.0) or (d_match > 0):
                current_score = base_score + total_boost
            else:
                # Penalty if no graph connection exists
                current_score = base_score * no_overlap_penalty

            # 4. TITLE JUNK PENALTY (Sequel Crusher)
            # If title is too similar, apply 0.9 reduction (crushing the score)
            cand_title = id_to_title.get(mid, "")
            is_sequel = False
            for s_title in seed_titles:
                if s_title in cand_title or cand_title in s_title:
                    # Basic check: only penalize if titles aren't identical (which are filtered anyway)
                    if s_title != cand_title:
                        is_sequel = True
                        break
            
            if is_sequel:
                # We use 0.1 (a 90% reduction) to align with your "0.9 penalty" intent
                current_score *= (1.0 - 0.9) 

            new_hybrid_scores.append((mid, current_score))
        
        final_scores = new_hybrid_scores

    # Final ranking
    print('Scoring and returning top-k results...')
    
    # Remove seeds from results
    final_scores = [t for t in final_scores if t[0] not in seed_ids]
    topk = sorted(final_scores, key=lambda x: -x[1])[:args.top_k]

    # Print results
    results = []
    id_to_display_title = {row['movieId']: row['title'] for _, row in movies_df.iterrows()}
    for mid, score in topk:
        results.append({'movieId': mid, 'title': id_to_display_title.get(mid, 'N/A'), 'score': score})

    print(json.dumps({'query_seeds': seed_ids, 'keywords': keywords, 'results': results}, indent=2))
    return results


def run_recommendation_logic(seed_titles, n_candidates=100, gnn_alpha=1.2, graph_rerank=True, top_k=10, user_boost=3.0, user_rerank=True, person_boost=2.0, no_overlap_penalty=0.9):
    class Args:
        def __init__(self):
            # Provided by API
            self.seed_titles = seed_titles
            self.n_candidates = n_candidates
            self.gnn_alpha = gnn_alpha
            self.graph_rerank = graph_rerank
            
            # Missing Defaults - Required by main()
            self.seed_ids = None
            self.keywords = None
            self.top_k = top_k
            self.use_gnn = True # Set True if you want the actual GNN model to run
            self.index_path = str(ROOT / 'movie_faiss.index')
            
            # Saved Tuning Parameters
            # To see more "Hidden Gems" and punish "Title Junk"
            self.person_boost = person_boost
            self.user_boost = user_boost
            self.user_rerank = user_rerank
            self.no_overlap_penalty = no_overlap_penalty
            self.heuristic_weight = 1.0
            
            # Model Architecture Defaults
            self.in_channels = 386
            self.hidden_channels = 128
            self.out_channels = 64
            self.num_layers = 2
            self.dropout = 0.1
            self.lazy_init = False
            self.max_user_neighbors = 1000
            self.max_person_neighbors = 500

    args = Args()
    results = main(args)

    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed_ids', type=str, default=None, help='Comma separated movieIds')
    parser.add_argument('--seed_titles', type=str, default=None, help='Comma separated movie titles (substring match)')
    parser.add_argument('--keywords', type=str, default=None, help='Comma separated keyword:weight pairs')
    parser.add_argument('--n_candidates', type=int, default=200, help='Number of FAISS candidates')
    parser.add_argument('--top_k', type=int, default=10, help='How many recommendations to return')
    parser.add_argument('--use_gnn', action='store_true', help='Whether to run GNN re-ranking (experimental)')
    parser.add_argument('--gnn_alpha', type=float, default=1.0, help='Mixing weight for GNN score vs content score (1.0=GNN only, 0.0=content only)')
    parser.add_argument('--max_user_neighbors', type=int, default=1000, help='Max number of user neighbors to include in subgraph')
    parser.add_argument('--max_person_neighbors', type=int, default=500, help='Max number of person neighbors to include in subgraph')
    parser.add_argument('--index_path', type=str, default=str(ROOT / 'movie_faiss.index'), help='Path to FAISS index')

    # model args (for loading GNN if used)
    parser.add_argument('--in_channels', type=int, default=386)
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--out_channels', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lazy_init', action='store_true', help='Use lazy initialization for SAGEConv layers to adapt to checkpoint dims')
    parser.add_argument('--graph_rerank', action='store_true', help='Use heuristic graph re-ranker (person Jaccard) as fallback or standalone')
    parser.add_argument('--user_rerank', action='store_true', help='Also use user-overlap (shared raters) as an additional heuristic signal')
    parser.add_argument('--person_boost', type=float, default=5.0, help='Additive boost multiplier for person Jaccard overlap')
    parser.add_argument('--user_boost', type=float, default=3.0, help='Additive boost multiplier for user Jaccard overlap')
    parser.add_argument('--no_overlap_penalty', type=float, default=0.7, help='Multiplier applied to content score when no overlap exists')
    parser.add_argument('--heuristic_weight', type=float, default=1.0, help='Global scaling weight for heuristic boosts')

    args = parser.parse_args()
    main(args)
