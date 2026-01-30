# API Optimizer
**An Automatic Cost, Performance, and Redundancy Optimization Layer for LLM API Usage**

API Optimizer is a runtime and static-analysis system that transparently optimizes Large Language Model (LLM) API usage by eliminating redundant calls, reusing semantically equivalent responses, and intelligently selecting cheaper models when task complexity allows, without requiring changes to application code.

The core idea behind API Optimizer is that LLM API calls, when used at scale, resemble database queries or compiler intermediate representations more than simple function calls. As such, they are amenable to classic systems optimizations such as caching, cost modeling, static analysis, and query planning, techniques that are largely missing from today’s LLM tooling.

---

## Motivation & Problem Statement

As LLMs move from experimentation into production systems, API usage often becomes:
- repetitive (same intent, slightly different phrasing),
- over-provisioned (high-cost models used for simple tasks),
- opaque (little insight into where cost and latency come from).

Unlike mature systems such as databases or compilers, most LLM applications lack:
- visibility into repeated or semantically equivalent prompts,
- automatic reuse of equivalent results,
- cost-aware model selection,
- tooling to reason about API usage at scale.

This results in:
- unnecessary cost from redundant prompts,
- increased latency from avoidable heavy model usage,
- no feedback loop between usage patterns and optimization.

API Optimizer explores how classic systems ideas (caching, static analysis, semantic equivalence, and cost modeling) can be applied to modern AI infrastructure.

---

## High-Level Idea

API Optimizer treats LLM API calls as optimization targets.

It introduces an intermediate optimization layer that:
1. observes API calls (statically and dynamically),
2. infers prompt intent and task complexity,
3. eliminates redundant work,
4. minimizes cost without sacrificing correctness.

Crucially, existing application code does not need to be rewritten or refactored.

---

## Architecture Overview

```
┌──────────────────┐
│  Your App (TS)   │
└────────┬─────────┘
         │ (LLM API call)
         ▼
┌─────────────────────────────┐
│ APIOptimizerWrapper         │
│                             │
│ 1) Exact cache lookup       │
│ 2) Semantic cache lookup    │
│ 3) Complexity classification│
│ 4) Model recommendation     │
│ 5) Token & cost estimation  │
└────────┬────────────────────┘
         │ (optimized request)
         ▼
┌─────────────────────────────┐
│ LLM Provider SDK            │
│ (OpenAI / Anthropic / etc.) │
└─────────────────────────────┘
```
---

## Core Components

### Runtime API Interception

API Optimizer monkey-patches the LLM client at runtime, intercepting calls to `chat.completions.create`.

**Why monkey-patching?**
- Zero-friction adoption
- No wrapper APIs to migrate to
- Mirrors how profilers, tracers, and debuggers are deployed in production systems

---

### Exact + Semantic Response Caching

The system implements a two-tier caching strategy.

#### Exact Cache
- Hashes prompt + model + parameters
- Avoids re-issuing identical requests
- TTL-based eviction prevents stale responses

#### Semantic Cache
- Uses sentence embeddings and cosine similarity
- Reuses responses across paraphrased prompts
- Configurable similarity thresholds preserve correctness

---

### Task Complexity Classification

API Optimizer analyzes prompts to infer task intent, such as:
- classification,
- extraction,
- summarization,
- reasoning,
- code generation.

Classification relies on:
- keyword and structural heuristics,
- prompt length and formatting signals,
- conservative confidence thresholds.

This enables safe downstream optimizations.

---

### Intelligent Model Recommendation

Based on inferred task complexity, the optimizer recommends or applies cheaper models when appropriate.

Examples:

| Task Type        | Example                         | Recommended Model |
|------------------|----------------------------------|-------------------|
| Classification   | Sentiment, labeling              | gpt-4o-mini      |
| Extraction       | Lists, parsing                   | gpt-4o-mini      |
| Reasoning / Code | Debugging, architecture design   | Higher-tier models |

---

### Token & Cost Estimation

The system estimates cost by:
- using `tiktoken` for OpenAI models,
- falling back to heuristics for non-OpenAI providers,
- modeling both input and expected output tokens.

---

### Static Codebase Analysis

API Optimizer includes an AST-based static analyzer that:
- scans Python codebases for LLM API calls,
- extracts prompt templates and f-string patterns,
- identifies repeated prompt structures,
- surfaces optimization opportunities before runtime.

---

## Observability & Metrics

The optimizer tracks:
- total API calls,
- cache hit rate,
- model downgrades,
- estimated cost savings.

Example output:

```
APIOptimizer wrapper activated!
   - Caching: ON
   - Model optimization: ON
   - Auto-apply: OFF

Cache hit! Saved $0.0021

Model Optimization Available:
   Current: gpt-4
   Recommended: gpt-4o-mini
   Reason: Task appears simple (summarize)
   Confidence: 90%
   Savings: $29.85 input / $59.40 output per 1M tokens

Calling gpt-4 API...
Response received in 0.42s
```

---

### Aggregated Metrics & Statistics

The optimizer maintains internal counters that track optimization effectiveness
across the lifetime of the application. These metrics provide visibility into
system-wide behavior rather than individual requests.

Metrics can be printed at any time using:
```python
wrapper.print_stats()
```

Tracked metrics include:
* total API calls,
* cache hit count and hit rate,
* model downgrades,
* estimated cumulative cost savings.

Example output:
```
============================================================
API OPTIMIZER STATISTICS
============================================================
Total API calls: 12
Cache hits: 5
Cache hit rate: 41.7%
Model downgrades: 3
Estimated cost saved: $0.0874

Cache entries: 5 valid, 0 expired
============================================================
```

These aggregated metrics enable developers to reason about cost, redundancy, and optimization effectiveness at the system level rather than on a per-request basis.

---

## Testing Strategy

The project includes:
- unit tests for semantic similarity and clustering,
- cache correctness tests (TTL, parameter sensitivity),
- runtime wrapper tests using mocked API clients,
- static analyzer tests for AST extraction.

---

## Future Improvements

### Learning-Based Token & Cost Models
Train lightweight regression models on real prompt/response pairs to replace heuristic response token estimation and improve cost prediction accuracy over time.

### Persistent Metrics Store
Store historical cost and cache metrics in SQLite or DuckDB to enable long-term trend analysis and reporting.

### Distributed Caching
Support Redis-backed exact and semantic caches for multi-instance or microservice deployments.

### Provider-Agnostic Abstraction Layer
Unify tokenization, pricing, and model metadata across OpenAI, Anthropic, and local LLMs.

### Prompt Canonicalization
Automatically normalize prompts (whitespace, formatting, structure) to improve cache hit rates without relying on embeddings.

### Feedback-Driven Optimization
Use observed runtime behavior to refine task classification thresholds and model recommendation confidence over time.

---

## License

MIT

