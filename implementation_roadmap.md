# Implementation Roadmap: Multi-Agent RAG Architecture

This document outlines the step-by-step implementation plan for the new multi-agent architecture, memory management, and scalability improvements.

## Phase 1: API Key Provisioning & Async Foundations
**Goal:** Remove hardcoded LLM dependencies, allow dynamic API key injection from the frontend, and ensure the core event loop is fully asynchronous.

*   **Step 1.1: Request Header Updates**
    *   Modify FastAPI endpoints (`/api/chat`, `/api/upload`) in `app.py` to accept a custom header (e.g., `X-LLM-API-Key`).
    *   Update `ChatRequest` and `UploadResponse` Pydantic models if necessary.
*   **Step 1.2: Dynamic Client Instantiation**
    *   Refactor `components/router.py` and `components/generation.py`. Remove the global `genai.configure(api_key=GOOGLE_API_KEY)` calls.
    *   Update the functions to instantiate the Gemini model *per-request* using the API key passed from the router/generation pipeline.
*   **Step 1.3: Asynchronous I/O Wrappers**
    *   Refactor `database/db_manager.py`. Wrap synchronous Pinecone calls (`query`, `upsert`, `delete`) in `asyncio.to_thread()`.
    *   Refactor `integrations/s3/client.py` and `integrations/huggingface/client.py` to use asynchronous HTTP requests (`httpx.AsyncClient` or `aiohttp`) instead of synchronous blocking calls.
    *   Update `integrations/duckduckgo/client.py` to run web searches asynchronously.

## Phase 2: Multi-Agent Orchestrator (LangGraph Integration)
**Goal:** Replace the monolithic procedural router with a modular, graph-based multi-agent supervisor system.

*   **Step 2.1: LangGraph Setup & State Definition**
    *   Define a global agent state schema (e.g., `AgentState` using `TypedDict`) containing the original query, intermediate retrieved chunks, web snippets, current API key, and routing decisions.
*   **Step 2.2: Sub-Agent Implementation**
    *   Create `agents/web_search_agent.py`: Asynchronously fetches DDG results and formats them.
    *   Create `agents/vectordb_agent.py`: Asynchronously embeds the query and searches Pinecone.
    *   Create `agents/context_synthesis_agent.py`: Takes the populated state and generates the final response.
*   **Step 2.3: Supervisor/Router Node**
    *   Create a Supervisor Agent that evaluates the query and determines which Sub-Agents to invoke (similar to the current `route_query` but outputting graph edges).
*   **Step 2.4: Graph Compilation**
    *   In `app.py`, compile the LangGraph workflow. Replace the linear `route -> check -> retrieve -> generate` sequence with a single invocation of the compiled graph.

## Phase 3: Advanced Memory Management
**Goal:** Implement a dual-memory system combining VectorDB (Pinecone) for semantics and a Knowledge Graph (Neo4j/NetworkX) for relational persistence.

*   **Step 3.1: Hierarchical Markdown Memory**
    *   Create a mechanism to read/write deterministic state to a persistent `GEMINI.md` or `memory.md` file per session/project.
    *   Prepend this loaded markdown context to the Supervisor Agent's prompts.
*   **Step 3.2: Knowledge Graph Setup**
    *   Integrate a graph database (e.g., NetworkX for in-memory MVP, or Neo4j for production).
    *   Create `database/graph_ops.py` to handle CRUD operations for nodes (Entities, Documents) and edges (Relationships).
*   **Step 3.3: Ingestion Pipeline Update**
    *   Modify `process_file_pipeline` in `components/preprocessing.py`. After generating standard vector chunks, use an LLM extraction step to identify entities and relationships.
    *   Save vectors to Pinecone *and* entities/relationships to the Knowledge Graph concurrently.
*   **Step 3.4: Graph Retrieval Integration**
    *   Update the `vectordb_agent` (now perhaps a general `memory_agent`) to query both Pinecone (for semantic text chunks) and the Knowledge Graph (for structured relational facts).

## Phase 4: Rich Frontend Output Formatting & Tooling
**Goal:** Enhance the LLM output to include interactive components and ensure the backend is horizontally scalable.

*   **Step 4.1: Structured Prompt Engineering**
    *   Update the `Context Synthesis Agent`'s system prompts. Instruct it to output data using robust Markdown (including image references `![alt](url)`) and structured JSON or custom HTML wrappers for interactive components.
*   **Step 4.2: Primary Operational Store Migration**
    *   Remove reliance on Pinecone's dummy vector queries for session management.
    *   Introduce PostgreSQL (or SQLite/MongoDB) to track active sessions, uploaded S3 keys, and authentication states. Update `api/cleanup` to use this relational DB.
*   **Step 4.3: Scalable Ingestion Queue**
    *   Replace FastAPI's `BackgroundTasks` with a dedicated task queue (Celery + Redis) for the `process_file_pipeline` to prevent the web server from being bogged down by CPU-heavy embedding tasks.
