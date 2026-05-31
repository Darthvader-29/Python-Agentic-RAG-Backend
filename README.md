# Python-Agentic-RAG-Backend

VectorDB: Pinecone | File Upload: AWS S3 | LLM: Gemini 2.5 Flash | Deployment: Render

## Quick start

```bash
cp .env.example .env   # fill in required values (see Phase 3 section below)
uv sync
uv run uvicorn app:app --host 0.0.0.0 --port 8000
```

## Auth flow (Phase 3)

All three RAG endpoints (`/api/chat`, `/api/upload`, `/api/cleanup`) require a valid **JWT access token**
in the `Authorization: Bearer <token>` header.

### Register / login / refresh

```bash
# Register
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"you@example.com","username":"you","password":"yourpass"}'

# Login — returns access + refresh tokens
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"you@example.com","password":"yourpass"}'

# Refresh access token
curl -X POST http://localhost:8000/auth/refresh \
  -H "Content-Type: application/json" \
  -d '{"refresh_token":"<your-refresh-token>"}'
```

### Store a BYOK LLM key

```bash
curl -X POST http://localhost:8000/api/keys \
  -H "Authorization: Bearer <access-token>" \
  -H "Content-Type: application/json" \
  -d '{"provider":"gemini","api_key":"AIza..."}'
```

Keys are encrypted at rest with Fernet. The plaintext never touches the database or logs.

## Phase 3 environment variables

| Variable | Required | Description |
|---|---|---|
| `JWT_SECRET` | **yes** | JWT signing secret — keep strong, never commit |
| `LLM_KEY_ENCRYPTION_KEY` | **yes** | Fernet key (base64, 32 bytes) for BYOK key encryption |
| `CORS_ALLOWED_ORIGINS` | **yes** | JSON list of allowed CORS origins, e.g. `["http://localhost:3000"]` |
| `JWT_ALGORITHM` | optional | Default: `HS256` |
| `ACCESS_TOKEN_TTL_MINUTES` | optional | Default: `15` |
| `REFRESH_TOKEN_TTL_DAYS` | optional | Default: `7` |

Generate a Fernet key:
```bash
uv run python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

## Running tests

```bash
uv run pytest                   # all tests (coverage gate: 67%)
uv run pytest --cov             # with coverage report
```

Tests that hit the real DB require `TEST_DATABASE_URL` in the environment; they are skipped otherwise.

## Commands

```bash
uv run uvicorn app:app --host 0.0.0.0 --port 8000
uv run pytest --cov
uv run ruff format . && uv run ruff check . --fix
uv run mypy app.py config.py exceptions.py dependencies.py auth database
uv run alembic upgrade head
uv run alembic downgrade -1
```
