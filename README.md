# GNN Movie Graph ETL

This workspace contains scripts to download public MovieLens and IMDb datasets and to build a unified schema for heterogeneous graph construction.

Quick start (Windows PowerShell):

1. Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\pip.exe install -r requirements.txt
```

2. Download datasets:

```powershell
.\.venv\Scripts\python.exe scripts\download_datasets.py
```

3. Run ETL to produce processed CSVs:

```powershell
.\.venv\Scripts\python.exe scripts\etl_schema.py
```

Notes:
- For plot summaries and richer metadata you can optionally enrich with TMDb/OMDb APIs (API keys required).
- Embedding/model packages (PyTorch, sentence-transformers, PyG/DGL) are intentionally left out of the minimal install and will be added when we start model training.



No need to download the datasets just the FAISS and Best model of GNN for wor

## Frontend

A simple Vite + React app is included at `frontend/my-react-app` for quick exploration. To run it:

```powershell
cd frontend/my-react-app
npm install
npm run dev
```

Create `frontend/my-react-app/.env` from `frontend/my-react-app/.env.example` and add your `VITE_TMDB_API_KEY` there. **Do not commit your `.env` file.**

## Notes on models & install

- The project uses PyTorch and PyTorch Geometric for GNN work; installing PyG may require platform-specific commands (see https://pytorch-geometric.readthedocs.io).
- FAISS is used for nearest-neighbor retrieval — `faiss-cpu` is included in `requirements.txt` for CPU installs.

---

*If you want, I can create a GitHub repository for you (requires GitHub account & auth) and push these changes publicly — tell me if you'd like me to proceed and whether you want the repo name to match the current folder (`GNN`) or a different one.*