# Chromio

A React + Python Flask full stack app for generating, building, and cataloging professional color palettes based on user preference or query.

## Running the App (Locally)

First open Docker Desktop

### Backend

```bash
cd backend
make docker-build-dev   # Build the dev Docker image (first time or after dependency changes)
make docker-dev          # Run the backend dev server
make run
```

### Frontend

```bash
cd frontend
make docker-build        # Build the dev Docker image (first time or after dependency changes)
make docker-dev              # Run the frontend dev server
make run_dev
```
