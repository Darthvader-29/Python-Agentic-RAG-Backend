# Implementation Roadmap

This document provides the chronological execution plan for implementing the Multi-Agent RAG architecture.

## Phase 1: API Key & Async Foundations
**Objective:** Prepare the backend infrastructure for high concurrency and dynamic configuration.
1.  **Request Headers:** Update FastAPI endpoints to accept `X-LLM-API-Key`.
2.  **Dynamic Clients:** Refactor `components/router.py` and `components/generation.py` to instantiate Gemini models per-request using the injected key.
3.  **Async Wrappers:** Refactor `database/db_manager.py`, `integrations/s3/client.py`, and `integrations/duckduckgo/client.py` to utilize `asyncio.to_thread` or native async HTTP clients (`httpx`).
4.  **Relational DB Migration:** Implement PostgreSQL/SQLite for session and file tracking, removing operational state management from Pinecone.

## Phase 2: Multi-Agent Orchestrator
**Objective:** Replace the monolithic router with a modular, graph-based agent supervisor system.
1.  **State Definition:** Define the `TypedDict` Agent State containing query, retrieved chunks, web snippets, and routing decisions.
2.  **Sub-Agent Creation:**
    *   Build `agents/web_search_agent.py`
    *   Build `agents/vectordb_agent.py`
    *   Build `agents/context_synthesis_agent.py`
3.  **Supervisor Node:** Implement the Supervisor Agent using LangGraph to evaluate queries and output parallel execution edges.
4.  **Graph Compilation:** Wire the nodes together in `app.py` and replace the legacy linear routing sequence.

## Phase 3: Advanced Memory Management
**Objective:** Introduce persistent markdown configurations and a relational knowledge graph.
1.  **Hierarchical Markdown:** Create loaders to read session-specific `memory.md` files and inject them into the Supervisor's prompt.
2.  **Graph DB Integration:** Set up NetworkX (MVP) or Neo4j.
3.  **Entity Extraction:** Update the ingestion pipeline to run an LLM pass over chunks to extract Nodes and Edges, saving them to the Graph DB.
4.  **Hybrid Retrieval:** Update the `vectordb_agent` to query both the VectorDB (semantic) and the Graph DB (relational).

## Phase 4: Frontend UI/UX Enhancements
**Objective:** Deliver rich, interactive output and reduce perceived latency.
1.  **Streaming:** Convert the `/api/chat` endpoint to use Server-Sent Events (SSE) to stream generation tokens and sub-agent status updates.
2.  **Rich Output Prompts:** Update the Synthesis Agent's system prompt to enforce Markdown usage, image linking, and structured JSON output for interactive UI components.
3.  **Ingestion Queueing:** Migrate document ingestion to a Celery/Redis message queue to ensure horizontal scalability under high load.