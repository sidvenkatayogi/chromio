# Chromio (ML)

### The most interesting part of this project is in /ml and /experiments. The full-stack app is more of a by-product of the ML work we did.

### The main thing we did was design and execute an RLVR fine-tuning protocol to improve color palette generation in small open-source LLMs. Used deterministic reward metrics for aesthetic output, and serverless LoRA fine-tuning on the Fireworks.ai platform


### Improved Qwen3-8B palette similarity scores by 13% over baseline, beating gpt-5-mini and gpt-4o!

### Need to do a write-up of this project eventually because it was really interesting, but basically we used this [paper](https://arxiv.org/abs/2508.08987) + fine-tuning to get SOTA performance.

# Chromio (Full-stack App):

A React + Python Flask full stack app for generating, building, and cataloging professional color palettes based on user preference or query.

## Prerequisites

- **Python 3.12+**
- **Node.js** (LTS recommended)
- **Docker Desktop** (if running via Docker)
- An **OpenAI API key** (used for ChromaDB embeddings and gpt-5-mini)
- A **Supabase** project (for authentication)

## Getting Started from a Fresh Clone

```bash
git clone <repo-url>
cd chromio
```

### 1. Backend Setup

```bash
cd backend
```

**Create your environment file:**

```bash
cp example.env .env
```

Edit `.env` and fill in your credentials:

```
OPENAI_API_KEY=your-openai-api-key
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-supabase-anon-key
```

#### Option A: Run with Docker (recommended)

```bash
make docker-build-dev   # Build the dev Docker image (first time or after dependency changes)
make docker-dev          # Start the backend dev container
```

#### Option B: Run locally with a virtualenv

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
make run                 # Starts Flask on http://localhost:5000
```

The backend API will be available at **http://localhost:5000/api/v1/**.

---

### 2. Frontend Setup

```bash
cd frontend
```

**Create your environment file:**

```bash
cp example.env .env.local
```

The default `VITE_SERVER_URL=http://localhost:5000` should work for local development. Change it if your backend runs elsewhere.

#### Option A: Run with Docker

```bash
make docker-build-dev    # Build the dev Docker image (first time or after dependency changes)
make docker-dev          # Start the frontend dev container
```

#### Option B: Run locally

```bash
npm install
npm run dev              # or: make run_dev
```

The frontend will be available at **http://localhost:5173**.

---

## Project Structure

```
chromio/
├── backend/          # Flask API server
│   ├── api.py        # App entrypoint
│   ├── routes/       # API route blueprints
│   ├── controllers/  # Request handlers
│   ├── models/       # Data models
│   ├── db/           # Database managers (ChromaDB, Supabase)
│   ├── errors/       # Custom error responses
│   ├── middleware/    # Flask middleware
│   └── color_utils/  # Color manipulation & palette sorting
├── frontend/         # React + Vite + Tailwind UI
│   └── src/
│       ├── components/
│       ├── context/
│       └── utils/
|
|   # ml/ and experiments/ are artifacts from previous fine tuning, not used in chromio
├── ml/               # ML evaluation & grading tools
│   ├── evaluator/    # Model evaluation scripts
│   └── grader/       # Palette quality scoring
└── experiments/      # Standalone ml experiment scripts
```

## Environment Variables

| Variable | Where | Description |
|---|---|---|
| `OPENAI_API_KEY` | backend | OpenAI key for ChromaDB embedding functions |
| `SUPABASE_URL` | backend | Supabase project URL |
| `SUPABASE_KEY` | backend | Supabase anon/public API key |
| `VITE_SERVER_URL` | frontend | Backend API URL (default: `http://localhost:5000`) |
| `FIREWORKS_API_KEY` | ml/evaluator | Fireworks.ai key (only needed for ML evaluation) |

## Makefile Commands

### Backend (`cd backend`)

| Command | Description |
|---|---|
| `make run` | Run Flask dev server locally |
| `make docker-build-dev` | Build the dev Docker image |
| `make docker-dev` | Run the backend in a dev Docker container |
| `make docker-build` | Build the production Docker image |
| `make docker` | Run the production Docker container |

### Frontend (`cd frontend`)

| Command | Description |
|---|---|
| `make run_dev` | Install deps & start Vite dev server |
| `make build` | Production build |
| `make docker-build-dev` | Build the dev Docker image |
| `make docker-dev` | Run the frontend in a dev Docker container |
