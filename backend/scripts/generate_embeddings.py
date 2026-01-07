"""Feature engineering and embedding generation for heterogeneous graph nodes.

Outputs (data/features):
- movie_features.npy: Dense feature matrix for movies (concatenated plot embeddings + structured features)
- movie_feature_names.txt: Column names for movie features
- feature_metadata.json: Metadata about feature generation (model, dims, normalization)

Uses:
- Sentence-BERT for plot/synopsis text embeddings
- Numeric feature scaling (MinMaxScaler)
- GPU acceleration (CUDA if available)
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
FEATURES = ROOT / "data" / "features"
FEATURES.mkdir(parents=True, exist_ok=True)

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L12-v2"  # ~384 dims, fast
EMBEDDING_DIM = 384


def get_embedder():
    """Lazy load sentence-transformers model with GPU support."""
    try:
        import torch
        from sentence_transformers import SentenceTransformer
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        print(f"  torch.cuda.is_available() = {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        model = SentenceTransformer(EMBEDDING_MODEL, device=device)
        return model, device
    except ImportError:
        raise ImportError("sentence-transformers not installed. Run: pip install -r requirements.txt")


def embed_texts(texts, embedder, batch_size=512, device="cpu"):
    """Embed a list of texts using Sentence-BERT with GPU acceleration in streaming batches.

    Large batch sizes (512+) work well on GPU. Returns numpy array of shape (len(texts), embedding_dim).
    """
    texts = [str(t) if pd.notna(t) else "" for t in texts]
    n = len(texts)
    print(f"Encoding {n} texts with batch_size={batch_size} on {device}...")
    all_embs = []
    for i in tqdm(range(0, n, batch_size), desc="embedding_batches"):
        batch_texts = texts[i:i+batch_size]
        emb = embedder.encode(batch_texts, batch_size=len(batch_texts), show_progress_bar=False, convert_to_numpy=True, device=device)
        all_embs.append(emb)
    embeddings = np.vstack(all_embs).astype(np.float32)
    print(f"Embedding complete. Shape: {embeddings.shape}")
    return embeddings


def process_movie_features():
    """Generate movie node features: embeddings + numeric features.
    
    Uses GPU to embed all movies efficiently without sampling.
    """
    print("\n=== Processing Movie Features ===")
    movies = pd.read_csv(PROCESSED / "movies.csv")
    
    embedder, device = get_embedder()
    
    # For this demo, create synthetic plot summaries from title + runtime
    # In production, fetch from TMDb or Wikipedia
    movies['plot_text'] = movies['title'].fillna("") + " (" + movies['runtimeMinutes'].fillna(0).astype(str) + " min)"
    
    # Embed ALL movies using GPU with large batch size (no sampling)
    print(f"Embedding {len(movies)} movie plots on {device} (batch_size=512)...")
    plot_embeddings = embed_texts(movies['plot_text'].tolist(), embedder, batch_size=512, device=device)
    
    # Numeric features: year (normalized), runtime (log scale)
    numeric_features = []
    for idx, row in movies.iterrows():
        year = float(row['year']) if pd.notna(row['year']) else 2000.0
        runtime = float(row['runtimeMinutes']) if pd.notna(row['runtimeMinutes']) else 100.0
        
        # Normalize year to [0, 1] range (assume 1890-2030)
        year_norm = (year - 1890) / (2030 - 1890)
        
        # Log-scale runtime
        runtime_log = np.log1p(runtime)
        
        numeric_features.append([year_norm, runtime_log])
    
    numeric_features = np.array(numeric_features, dtype=np.float32)
    
    # Concatenate embeddings + numeric features
    movie_features = np.concatenate([plot_embeddings, numeric_features], axis=1)
    
    # Normalize all features to [0, 1]
    scaler = MinMaxScaler()
    movie_features = scaler.fit_transform(movie_features).astype(np.float32)
    
    # Save features and metadata
    np.save(FEATURES / "movie_features.npy", movie_features)
    
    feature_names = [f"plot_emb_{i}" for i in range(EMBEDDING_DIM)] + ["year_norm", "runtime_log"]
    with open(FEATURES / "movie_feature_names.txt", "w") as f:
        f.write("\n".join(feature_names))
    
    print(f"Saved movie features: {movie_features.shape}")
    return movies, movie_features


def main():
    print("=" * 60)
    print("Feature Engineering & Embedding Generation (GPU Optimized)")
    print("=" * 60)
    
    # Process movie features (core)
    try:
        movies, movie_features = process_movie_features()
    except Exception as e:
        print(f"Error in movie features: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Save metadata
    metadata = {
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": EMBEDDING_DIM,
        "num_movies": len(movies) if movies is not None else 0,
        "movie_feature_dim": movie_features.shape[1] if movie_features is not None else 0,
        "normalization": "MinMaxScaler [0, 1]",
        "note": "People and user features skipped for now; can be added later.",
    }
    
    with open(FEATURES / "feature_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Feature Engineering Complete")
    print("=" * 60)
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
