# Internal Agentic Architecture

## 1. The Monolithic Router Problem
Currently, the backend utilizes a monolithic `route_query` function that forces a rigid linear sequence: Route -> Check DB -> Retrieve -> Generate. As tasks become more complex (requiring multi-step research, tool execution, or combining web and vector data dynamically), this linear procedural loop becomes fragile and difficult to extend.

## 2. Multi-Agent Orchestrator Paradigm
Drawing inspiration from state-of-the-art frameworks like **OpenClaw** and the leaked **Claude Code Architecture**, we will implement a modular, graph-based Multi-Agent Orchestrator. The core philosophy is to treat the AI not as a single omniscient brain, but as a structured assembly line of isolated, highly specialized sub-agents governed by strict deterministic infrastructure.

### 2.1 The Supervisor Agent (The Router)
*   **Role:** The main entry point for the backend. It receives the user's query and the session context.
*   **Function:** Instead of generating the final answer, the Supervisor analyzes the query and delegates tasks to specialized sub-agents. It acts as the routing node in a directed graph (e.g., using **LangGraph**).
*   **Output:** Generates a routing plan and parallel execution directives.

### 2.2 Specialized Sub-Agents (The Daemons)
To ensure isolation and prevent context bloat, sub-agents are executed as restricted "daemons" or "swarms". They possess narrow system prompts and specific tool access.

*   **Web Search Agent:**
    *   *Purpose:* Fetches real-time internet data.
    *   *Tools:* Asynchronous DuckDuckGo client.
*   **VectorDB Retrieval Agent:**
    *   *Purpose:* Fetches semantic chunks related to private documents.
    *   *Tools:* Pinecone querying, HuggingFace embeddings.
*   **Knowledge Graph Agent:**
    *   *Purpose:* Queries relational and structured memory.
    *   *Tools:* Neo4j/NetworkX cypher queries.
*   **Context Synthesis Agent (The Generator):**
    *   *Purpose:* The final node in the graph. It aggregates the raw data outputted by the other sub-agents, applies user formatting preferences, and generates the final response sent to the frontend.

### 2.3 Asynchronous Execution & Safety
*   **Concurrency:** When the Supervisor dictates that a query requires both Web and Private Document data, the Web Search Agent and VectorDB Retrieval Agent are executed simultaneously using `asyncio.gather()`.
*   **Defense in Depth:** As seen in Claude Code, the actual "AI decision logic" should be minimal. The vast majority of the architecture should be deterministic infrastructure—permission gates, tool execution sandboxes, and context recovery loops—ensuring the sub-agents operate safely and predictably.

## 3. Tool Routing and the Agent Loop
The execution flow is fundamentally a state machine:
1.  **State Initialization:** The incoming query is placed into a `TypedDict` Agent State.
2.  **Supervisor Evaluation:** The Supervisor evaluates the State and updates it with task assignments.
3.  **Sub-Agent Execution:** The assigned tools execute asynchronously, appending their results (web snippets, vector chunks) to the State.
4.  **Synthesis:** The Synthesis Agent reads the fully populated State and generates the final output.