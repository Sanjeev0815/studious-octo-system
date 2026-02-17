# Multi-Level Semantic Matching in Graph-Based Expert Discovery

## Overview
This project builds an expert discovery system for community Q&A using semantic embeddings, multi-level similarity scoring, graph analytics, and learning-to-rank. The workflow follows the phases in the project request.

## Setup
1. Create a MySQL database (or allow the loader to create it).
2. Set environment variables (or create `expert_discovery/.env`):
   - `DB_HOST` (default `127.0.0.1`)
   - `DB_USER` (default `root`)
   - `DB_PASSWORD`
   - `DB_PORT` (default `3306`)
   - `DB_NAME` (default `expert_discovery`)
3. Install dependencies:
   - `pip install -r requirements.txt`

## Phase 1 - Database Setup
- Schema: `database/schema.sql`
- Load data:
  - `python database/load_data.py --apply-schema --reset`

## Phase 2 - Embeddings
- Generate embeddings:
  - `python embeddings/generate_embeddings.py`

## Phase 3/4 - Similarity + Graph
- Build graph and metrics:
  - `python graph/build_graph.py`

## Phase 6 - Ranking Model
- Create features and train:
  - `python ranking/train_ranker.py`

## Phase 7 - Streamlit App
- Run:
  - `streamlit run app/streamlit_app.py`

## Notes
- The default ranking formula is in `utils/similarity.py` and `ranking/feature_engineering.py`.
- The optional GNN phase is not implemented in code yet.
