# Advanced Memory and Context Management

## 1. The Limitations of Pure Semantic Search
Currently, the system relies exclusively on Pinecone for memory. While highly effective at retrieving semantically similar text chunks, Vector Databases struggle with:
*   **Relational Logic:** (e.g., "Which document did User A upload yesterday that references the contract uploaded last week?")
*   **Persistent Preferences:** Remembering strict deterministic rules (e.g., "Always reply in Spanish", "Never use emojis") across sessions.

To solve this, we will adopt a multi-layered memory architecture modeled after the **Gemini CLI** and the leaked **Claude Code Three-Layer Memory System**.

## 2. Layer 1: Hierarchical Markdown Memory (Deterministic Context)
To enforce strict rules and remember user preferences without clogging the VectorDB, we will implement flat-file markdown memory.

*   **Mechanism:** The system maintains persistent markdown files (e.g., `~/.agent/GEMINI.md` for global preferences, or session-specific `memory.md` files).
*   **Function:** These files store deterministic facts, coding styles, or architectural guidelines.
*   **Integration:** Before the Supervisor Agent evaluates a query, the backend reads the relevant `.md` memory files and injects their contents directly into the system prompt. This guarantees that foundational context is never "lost" in a vector search.

## 3. Layer 2: The Knowledge Graph (Relational Memory)
To complement the semantic power of Pinecone, we will introduce a Knowledge Graph (e.g., Neo4j, or an in-memory NetworkX graph for MVP).

*   **Concept:** A graph maps entities (Nodes) and their relationships (Edges).
*   **Ingestion Update:** During the file processing pipeline (`process_file_pipeline`), after text is chunked and embedded for Pinecone, an LLM extraction pass is run over the chunks to extract structured entities.
    *   *Example:* `(Document_A) -[CONTAINS]-> (Entity: Q3_Revenue)` and `(Entity: Q3_Revenue) -[INCREASED_BY]-> (Entity: 15%)`.
*   **Retrieval:** The `Knowledge Graph Agent` uses Cypher queries to navigate these relationships, providing the LLM with structured facts that a standard cosine-similarity search might miss.

## 4. Layer 3: Vector Database (Semantic Memory)
Pinecone remains the workhorse for dense, unstructured text retrieval.
*   **Role:** Finding exact paragraphs, clauses, or nuances within large documents.
*   **Optimization:** Chunks retrieved from Pinecone will be cross-referenced with the Knowledge Graph and the Markdown Memory to provide a highly compacted, deeply relevant context payload to the Synthesis Agent.

## 5. Context Compaction
As multi-agent interactions grow, the context window can quickly hit token limits. The system will implement **Context Compaction Stages**:
1.  Summarizing older conversational turns.
2.  Truncating web search snippets to only the most relevant sentences.
3.  Prioritizing Markdown Memory (highest priority) over deep semantic chunks (lower priority) when approaching token limits.