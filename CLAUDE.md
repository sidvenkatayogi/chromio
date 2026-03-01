# Project Overview

This is a React - Python Flask based full stack website. The purpose of this website is to generate, build, and catalog professional color palettes based on user preference or query. 

**YOU MUST**: Do not add external dependencies without approval. New dependencies must result in a manual docker instance rebuild.
**YOU MUST**: Never commit or add changes to git without my permission.

## Key Commands

### Backend: cd backend

- `make docker-dev` - docker run latest dev image
- `make docker-build-dev` - docker build latest image and requirements

### Frontend: cd frontend

- `make docker` - docker run latest dev image
- `make docker-build` - docker build latest image and dependencies
- `make build` - build for production
- `make run_dev` - run dev build

## Important Caveats

- For endpoints that call external API like server_url/api/v1/text2palette/ do not call them on mass unless for good reasons like testing
- All server_url must include 

---
paths:
- `backend/**/*.py`
---

## API Development Rules

- All API endpoints must include reasonable input validation
- Use the custom error response format within backend/errors/ before then attempting standard error response format
- Include OpenAPI documentation comments