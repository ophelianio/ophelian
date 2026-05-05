# RFC 0001 — Ophelian Agent: declarative agent graphs with a real compiler

- **Status:** Draft
- **Target:** v1.1 (additive, after PyPI v1.0.0)
- **Module:** `ophelian.agents` (new), reusing `ophelian.core.compiler`,
  `ophelian.envs.*`, `ophelian.observability.*`
- **Author:** Luis Falva
- **Last updated:** 2026-05-04

## 1. Problem

Production agent workloads (autonomous CRM/ERP loops, long-running
analyst agents, RAG-with-tools pipelines) suffer four cost multipliers
that no current stack addresses jointly:

1. **Per-call opacity.** Each LLM invocation is treated as an
   independent HTTP request. A 40-step ReAct loop with 90% prompt
   overlap recomputes the prefix 40 times.
2. **Model rigidity.** Switching between a strong reasoner and a cheap
   extractor inside the same loop requires manual orchestration and
   pays cold-start latency on every switch.
3. **Topology blindness.** The runtime has no view of the agent graph,
   so it cannot batch parallel tool calls, prefetch the next likely
   branch, or speculatively decode the next LLM call.
4. **Compute placement is hand-wired.** Every agent framework today
   assumes one inference endpoint. Multi-cloud / spot / cost-optimal
   placement is the developer's problem.

## 2. Honest landscape survey

We are not first. Anyone who claims "nobody is doing this" has not
read the literature. What exists:

| Project | What it does well | What it does not do |
| --- | --- | --- |
| **vLLM / TGI** | Per-request throughput, paged attention | No graph IR, no cross-call planning |
| **SGLang** | RadixAttention KV reuse via prefix tree, structured generation | Single-server, no multi-cloud, no cost router |
| **Mooncake (Kimi)** | Disaggregated prefill/decode, distributed KV pool | Inference layer only, no agent semantics |
| **LMCache** | KV cache persistence to disk/RAM across sessions | No graph view, no scheduler |
| **Modular / Mojo** | Compiler-first inference, tight kernels | Closed source, single-vendor, no agent IR |
| **Parrot (MSR)** | Semantic Variables as IR for LLM-app graphs | Research, not productized |
| **LangGraph / LlamaIndex / CrewAI** | Python orchestration ergonomics | No compiler, no KV optimization, no multi-cloud, no spot, no cost router |

Two clusters: **inference servers with deep optimization but no agent
DSL**, and **agent DSLs with great DX but zero runtime intelligence**.
Nobody owns both halves.

## 3. Ophelian's wedge (the only unique combination)

Ophelian already ships four primitives the agent space lacks:

1. **Declarative immutable IR** (`Pipeline`, `Train`, `Deploy` as
   Pydantic v2 models) — agents become another node type in the same
   graph, not a parallel framework.
2. **Graph compiler with `dry-run`** (`ophelian/core/compiler.py`) —
   already turns user code into an `ExecutionPlan`. Extend its IR to
   express LLM calls, tool calls, and branches with cache-affinity
   metadata.
3. **Multi-cloud provider abstraction** (`Standalone`, `AWS`, `GCP`,
   `Azure`) — an agent's individual LLM calls can land on different
   providers based on cost and capability, not just one endpoint.
4. **`Auto()` cost router with spot awareness + checkpoint/resume** —
   already routes ML training to the cheapest GPU; reusing this for
   agent placement is the moat. SGLang does not know about EC2 spot
   prices. LangGraph does not know what an A10 costs.

The unique sentence: **declarative agent graphs that compile once and
run cost-optimally across any provider, with KV-cache reuse, model
switching, and spot resilience handled by the framework, not the
developer**.

## 4. Proposed surface (sketch, not contract)

```python
from ophelian.core import Pipeline
from ophelian.agents import Agent, LLMCall, ToolCall, Branch, Memory
from ophelian.envs import Auto

agent = Agent(
    memory=Memory(kind="kv_cache", scope="graph"),
    steps=[
        LLMCall(
            name="planner",
            model="anthropic:claude-sonnet-4",
            prompt_template="planner.j2",
            cache_prefix="system+tools",
        ),
        Branch(
            decided_by="planner.tool_choice",
            branches={
                "search": [
                    ToolCall("search", cache_key="$query"),
                    LLMCall(
                        name="extractor",
                        model="self_hosted:llama-3.1-8b",
                        prompt_template="extractor.j2",
                    ),
                ],
                "respond": [],
            },
        ),
        LLMCall(
            name="responder",
            model="anthropic:claude-sonnet-4",
            cache_prefix="system+tools+history",
        ),
    ],
)

Pipeline([agent]).run(env=Auto(cheapest_gpu="A10", spot=True))
```

## 5. What the compiler does that competitors do not

The `AgentPlan` extends `ExecutionPlan` with five passes:

1. **Prefix-cache analysis.** Detect `cache_prefix` overlaps between
   `LLMCall` nodes statically; emit reuse hints the runtime honors via
   SGLang RadixAttention or equivalent on supported backends.
2. **Model-affinity placement.** Group LLM calls by model so a cold
   weights load is amortized across the graph; place tool-only steps
   on CPU nodes.
3. **Branch speculation.** Where a `Branch` decision distribution is
   skewed (learned from telemetry, see Task #28 ledger), prefetch the
   most likely branch's prefix into KV cache during the planner's
   decode.
4. **Cost-aware multi-provider placement.** Reuse the `Auto()` router:
   `claude-sonnet` calls hit Anthropic's API, `llama-3.1-8b` calls hit
   the cheapest available spot A10, all within one declarative graph.
   The $/run estimate from the cost ledger (Task #28) becomes a
   first-class compile-time output.
5. **Spot-resilient checkpointing.** Reuse the existing AWS spot
   auto-resume work (Task #2): agent state is checkpointed between
   LLM calls so an interrupted spot instance resumes mid-conversation
   on a new node, not from scratch.

None of these passes require a new runtime — they extend the existing
compiler and reuse providers.

## 6. Phased rollout

Each phase is one project task, shippable independently. No phase
blocks v1.0.0 PyPI publish.

- **v1.1 — Minimum viable agent.** `ophelian.agents` package with
  `Agent`, `LLMCall`, `ToolCall`, `Branch`. Single-provider execution
  (Anthropic only initially). Lifecycle events emitted per step (reuse
  Task #29 bus). No KV optimization yet. Validates the IR.
- **v1.2 — Multi-LLM cost router.** Extend `Auto()` to price LLM
  endpoints (Anthropic / OpenAI / self-hosted) per-token, route each
  `LLMCall` to the cheapest endpoint that satisfies declared
  constraints (`min_context`, `requires_tool_use`, etc.). Cost ledger
  (Task #28) records per-call spend.
- **v1.3 — KV cache reuse.** Compiler pass for prefix detection;
  runtime adapter for SGLang on self-hosted models. API-only models
  (Anthropic, OpenAI) get prompt caching when the provider supports
  it. Document the gap.
- **v1.4 — Spot-resilient agent state.** Reuse AWS spot resume to
  checkpoint agent state between steps; resume on interruption.
- **v2.0 — Distributed agent graphs.** Cross-node KV pool
  (Mooncake-style), branch speculation across nodes.

## 7. Anti-scope (what we will not build)

- **Not building a new inference server.** We integrate with vLLM /
  SGLang / API providers via thin adapters. Reinventing paged
  attention is a five-year detour.
- **Not building a vector DB.** RAG storage stays user's problem;
  we provide `ToolCall` integration points, nothing more.
- **Not building a prompt management UI.** Prompts are files in the
  user's repo, versioned by git, like every other code artifact in
  Ophelian.
- **Not adding a streaming-first protocol** in v1.1. Streaming
  belongs at the runtime adapter layer once the IR is stable.
- **Not adding agent-to-agent messaging** (multi-agent
  orchestration). One agent graph at a time. Multi-agent is v2+.

## 8. Open questions (must resolve before v1.1 implementation)

1. **Branch decision contract.** Does the planner return a structured
   tool choice (function-calling JSON) that the compiler validates,
   or a free-form string the user parses? Strong preference for
   structured.
2. **Memory scopes.** `scope="graph"` is obvious; do we need
   `scope="tenant"` for the SaaS pivot, and how does that interact
   with the cost ledger's `context.tenant_id`?
3. **Failure semantics.** If `extractor` 5xx's mid-graph, do we
   re-plan, retry with backoff, or fail the run? Default + override.
4. **API-provider prompt caching.** Anthropic supports prompt
   caching with explicit cache breakpoints; OpenAI auto-caches
   prefixes. Does the compiler emit provider-specific cache markers
   or a normalized hint the adapter translates?
5. **Token usage accounting.** Reuse `ophelian.serve.tokens.in/out`
   metrics or introduce `ophelian.agents.tokens.*`? Prefer reuse for
   dashboard continuity (Task #34).

## 9. Success criteria for v1.1

- A user can write a 30-line `Agent(...)` definition and run it
  end-to-end against Anthropic with one `Pipeline([agent]).run(env=...)`.
- The same agent runs unchanged on `Standalone(local=True)` for
  development and `AWS(...)` for production.
- Lifecycle events fire per `LLMCall` and `ToolCall` (reusing the
  Task #29 bus).
- Cost ledger (Task #28) records one row per agent run with token
  totals.
- `ophelian dry-run` produces a readable plan showing each step,
  chosen model, estimated cost, and cache-reuse opportunities.
- Documentation: one tutorial, one cookbook recipe, one architecture
  doc explaining the IR.
