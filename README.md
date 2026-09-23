### Agentic Feature

**Feature added:** Cross-source verification for multi-paper research comparison.

The Week 15 assistant was extended with a new `app/agent.py` module. The existing RAG system in `app/rag.py` was also extended to support source-specific retrieval from multiple research papers. The agent can compare the available evidence, identify missing sources, and request additional retrieval before generating the final answer.

### Cross-Source Verification

The agentic feature added to the Week 15 assistant is **cross-source verification**. The assistant was extended to work with multiple research papers and verify whether sufficient evidence has been collected from the required sources before answering.

### Agentic Loop

The agent follows an iterative loop:

```text
User Question
      ↓
Initial RAG Retrieval
      ↓
Evaluate Retrieved Evidence
      ↓
Is Evidence Sufficient?
   /              \
 No                Yes
 ↓                  ↓
Select Missing     Generate
Source             Final Answer
 ↓
Source-Specific Retrieval
 ↓
Evaluate Again
 ```
### Why a Fixed Pipeline Is Not Sufficient

A fixed pipeline is insufficient because the assistant must evaluate the retrieved evidence and dynamically decide whether additional sources need to be searched before generating the final answer.


### a. Context Engineering Technique

**1. Which technique was used:**  
The implementation uses **retrieval capping and context compaction**.

**2. Where it is applied in the agentic loop:**  
After each retrieval step, the retrieved evidence is passed through `compact_context()` in `app/agent.py`. The system keeps only one chunk per source and limits each chunk to 1,600 characters before the evidence is given to the agent. When the agent requests a specific missing paper, source-specific retrieval is also used to add only relevant evidence.

**3. What problem it solves:**  
Because the agent can perform multiple retrieval iterations, evidence from previous searches could accumulate and make the LLM context increasingly large. This previously caused requests to exceed the model's token limit. Context compaction was therefore introduced after each retrieval step to keep the context bounded, reduce token usage, and prevent oversized LLM requests while preserving evidence from each required research paper.

### b. Agentic Pattern

**1. Pattern used:**  
The implementation uses a **single-agent loop**.

**2. Why this pattern was chosen:**  
A single-agent design is appropriate because the task requires one reasoning process to evaluate the retrieved evidence, identify missing research papers, decide whether another search is necessary, and determine when the evidence is sufficient. Using multiple agents would introduce additional coordination and communication overhead without providing a clear benefit for this focused research-comparison task.

The agent follows this iterative process:

```text
Retrieve → Evaluate Evidence → Decide → Search Again / Finish
```

### c. Evaluation Harness

A custom evaluation harness was built from scratch in
`evaluation/evaluate_agent.py` to test the actual agentic workflow.

The harness runs multiple research queries through the real agent and measures:

1. **Task completion rate** – percentage of queries for which the agent
   successfully reaches a sufficient-evidence state and generates a
   final answer.

2. **Tool-call correctness** – checks whether `search_again` decisions
   contain a valid search query and a valid target research paper.

3. **Trajectory length** – records the number of agent iterations required
   for each query and calculates the average trajectory length.

4. **Failure log** – records unsuccessful cases and classifies them as
   hard failure, soft failure, or cascading soft failure.

The evaluation results are saved to `evaluation/results.json`.

The harness runs test queries sequentially to avoid unnecessary API
rate-limit pressure and evaluates the actual agent decisions rather than
a separate mock implementation.

### Skill vs. Agent

This capability could not be effectively implemented as a Skill because the assistant must dynamically evaluate retrieved evidence and decide whether to search another source.

A Skill would provide predefined instructions for a known workflow, whereas our agentic feature requires runtime decision-making based on the results of previous retrieval steps. Therefore, we used a **single-agent loop** instead of a Skill.

### 2. Token and Cost Accounting

The evaluation harness records token usage for every test query using the
token usage information returned by the LLM API. For each query, the
harness records input tokens, output tokens, and total tokens consumed
across the agent's evidence-evaluation and final-answer calls.

Because this implementation uses a single-agent loop, there is no
additional multi-agent coordination cost to compare. The per-query token
totals are used to show the cost of the agentic workflow and how token
usage changes with the number of agent iterations.

### 3. Failure Injection Test

**Injected failure:** Malformed retrieval output.

A malformed retrieval result was intentionally injected by removing the
required `source` and `text` fields from the retrieved evidence. The agent
validates retrieval results before using them in its reasoning loop.

The injected failure was detected as invalid evidence, causing the request
to stop with a **hard failure** rather than generating a confident answer
from incomplete evidence. The result was recorded in
`evaluation/failure_test_results.json`.

This demonstrates that the system detects malformed evidence instead of
silently continuing with unsupported information.

### 4. Tool vs. Agent Boundary

The assistant models document retrieval as a **bounded tool call** rather than an agent-to-agent interaction. Retrieval is a specific, stateless operation that returns relevant evidence, while the main agent decides when and what to search next. This keeps the architecture simple and gives the agent control over the overall reasoning process.

