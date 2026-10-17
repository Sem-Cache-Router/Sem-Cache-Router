## SemCache-Router

## A Caching and Reliability Gateway for LLM Provider APIs

Document Product Requirements Document Version 1.0 Track Generative AI Domain AI Infrastructure

Project type OJT Self Assign Duration 12 weeks Team size 2 Skill level Intermediate

Primary users Application developers, platform engineers,

engineering managers

Constraint Provider API cost budget Institution Polaris School of Technology Repository github.com/Anirudh0465

Date 30 August 2026


## Contents

- 1. Executive Summary

- 2. Problem Statement

- 3. Objectives

- 4. Personas

- 5. Business Requirements

- 6. Non Functional Requirements

- 7. Product Goal and Feature Priorities

- 8. User Stories

- 9. Product States

- 10. Product Principles

- 11. Scope

- 12. Technical Requirements

- 13. High Level Design

- 14. Request Lifecycle

- 15. Cache Design

- 16. Rate Limiter Design

- 17. Circuit Breaker Design

- 18. Database and Data Design

- 19. API Specification

- 20. Low Level Design

- 21. Evaluation Methodology and Acceptance Criteria

- 22. Success Metrics

- 23. Risks

- 24. Testing Strategy

- 25. Observability

- 26. Security Design

- 27. Deployment Architecture

- 28. Cost Analysis

- 29. 12 Week Project Roadmap

- 30. Architecture Decision Records

- 31. Traceability Matrix

- 32. Future Scope

- 33. Implementation Amendments


## SemCache-Router

Product Requirements Document, version 1.0

## 1. Executive Summary

SemCache-Router is a gateway service that sits between an application and one or more large language model provider APIs. It exposes an OpenAI compatible endpoint, so an existing application can point at it by changing a base URL and nothing else. Behind that endpoint it adds the infrastructure layer that raw provider calls do not have: a two tier response cache, a token aware rate limiter, and failover across providers when one of them degrades.

The project is deliberately scoped around measurement rather than features. Anyone can build a wrapper that forwards a request to OpenAI. The claim this project makes is narrower and testable: that a semantic cache in front of an LLM API reduces cost and latency by a measurable amount, at a quality level that can be reported honestly, and that the cost of a provider outage can be reduced to zero user visible errors. Each of those claims has an experiment attached to it, and each experiment can fail.

The system is built in Python with FastAPI, using Redis for the exact match cache and the rate limiter, and ChromaDB as the vector store for the semantic cache. The whole stack runs under Docker Compose so that a benchmark run is reproducible on any machine.

## 2. Problem Statement

Calling an LLM provider API directly is straightforward to write and expensive to operate. Three specific costs show up as soon as an application has real traffic.

Redundant inference. Users ask the same question in different words. "What is your refund policy" and "how do I get my money back" are, for most support applications, the same query with the same correct answer. A naive client pays full token cost for both, every time, forever. Exact string caching catches almost none of this, because natural language rarely repeats verbatim.

Rate limit failures. Providers enforce limits on tokens per minute, not requests per minute. An application that tracks request counts will still hit 429 responses, because a handful of long prompts can exhaust a token budget that a hundred short ones would not. Handling this correctly requires estimating token cost before the request is sent, not after it fails.

Single provider dependency. When a provider has an incident, an application with one hardcoded client goes down with it. Retrying immediately makes this worse: every client retries at the same moment, and the provider receives a synchronised burst of traffic exactly when it is least able to serve it.

These are not novel problems and they are not LLM specific. Caching, admission control and circuit breaking are standard distributed systems concerns. What is specific here is that semantic similarity, rather than key equality, decides a cache hit, and that introduces a correctness question that a normal cache does not have: a cache hit can be wrong. Measuring how often it is wrong, and at what threshold, is the central technical problem of this project.

## 3. Objectives

- Provide a drop in OpenAI compatible endpoint that requires no application code changes beyond a base URL.

- Reduce spend on repeated and near repeated queries through a two tier cache, and quantify the reduction on a fixed public workload.

- Quantify the accuracy cost of semantic caching at several similarity thresholds, and report the trade off rather than a single flattering number.

- Enforce a token budget per API key that behaves predictably regardless of cache state.

- Survive the failure of a primary provider without surfacing errors to the caller.

- Instrument every request so that cost, latency, cache status and breaker state are observable rather than inferred.

## 4. Personas

| Persona | Goal | Pain point today |
| --- | --- | --- |
| Application developer Add caching and failover without rewriting the |   | Every provider client is different, and reliability logic ends up |
|   | application | duplicated in application code |
| Platform or | Enforce spend limits across several teams | No central place to apply a token budget, so one team can |
| infrastructure | sharing one provider account | exhaust the quota for everyone |
| engineer |   |   |
| Engineering manager Understand and forecast LLM spend |   | Provider dashboards report totals, not which queries were |
|   |   | avoidable |

## 5. Business Requirements

| ID | Requirement | Priority | Acceptance criteria |
| --- | --- | --- | --- |


| BR-001 | Accept chat completion requests on an OpenAI | Must | An unmodified OpenAI SDK client works against the |
| --- | --- | --- | --- |
|   | compatible endpoint |   | gateway with only a base URL change |
| BR-002 | Serve an identical repeat prompt from cache | Must | Second identical request returns a cached response and |
|   | without calling a provider |   | records zero provider cost |
| BR-003 | Serve a semantically similar prompt from cache | Must | A paraphrase above the threshold returns a cached |
|   | above a configured threshold |   | response with its similarity score recorded |
| BR-004 | Enforce a token budget per API key | Must | A caller over budget receives 429 with a retry hint, |
|   |   |   | independent of cache state |
| BR-005 | Expire cache entries on a configurable TTL | Must | An entry past its TTL is not served and is removed from |
|   |   |   | both tiers |
| BR-006 | Report cost, latency and hit ratio for a workload | Must | Benchmark harness produces a comparable report |
|   | with caching on and off |   | across at least three thresholds |
| BR-007 | Report the accuracy of cached answers against | Must | A quality score per threshold is produced and plotted |
|   | fresh answers |   | against hit rate |
| BR-008 | Fail over to a secondary provider when the | Should | Primary is failed during a live run and no caller visible |
|   | primary is unhealthy |   | error is emitted |
| BR-009 | Route simple prompts to a cheaper model | Should | Routing decision and chosen model are recorded per |
|   |   |   | request |
| BR-010 | Emit distributed traces for every request | Should | Spans carry cache status, similarity, cost and breaker |
|   |   |   | state |
| BR-011 | Pass streaming responses through unmodified | Could | An SSE client receives incremental tokens through the |
|   |   |   | gateway |

## 6. Non Functional Requirements

| Attribute | Target |
| --- | --- |
| Cache overhead | Tier 1 lookup under 5 ms at p95. Tier 2 lookup under 60 ms at p95 on the reference workload. |
| Throughput | Sustain 50 concurrent requests on a single container without queue growth, verified with Locust. |
| Durability | Cache and limiter state survive a container restart. A benchmark run started after a restart must not begin |
|   | cold unless explicitly reset. |
| Reproducibility | A benchmark run is reproducible from a committed workload file and a committed configuration. Reported |
|   | numbers include the git commit and the threshold used. |
| Configurability | Similarity threshold, TTL, bucket capacity and refill rate are configurable without a code change. |
| Degradation | If ChromaDB is unavailable the gateway continues serving with Tier 1 only and records the degradation, |
|   | rather than failing the request. |
| Secrets | Provider keys are read from environment only and never written to logs, traces or cache entries. |

## 7. Product Goal and Feature Priorities

The goal is a gateway that an application can adopt in one line and that produces evidence of its own value. Evidence means a report, generated by a committed script, that a reviewer can regenerate.

| Feature | Priority | Justification |
| --- | --- | --- |
| OpenAI compatible endpoint | Must | Without it, adoption requires an application rewrite and |
|   |   | the project is a demo rather than a gateway |
| Tier 1 exact match cache | Must | Cheapest possible hit, and the guard that stops literal |
|   |   | repeats from paying embedding cost |
| Tier 2 semantic cache | Must | The actual subject of the project |
| Token bucket rate limiter | Must | Budget enforcement, and the reason the gateway is |
|   |   | infrastructure rather than a cache library |
| TTL and eviction | Must | An unbounded cache serving stale answers is a |
|   |   | correctness bug, not a feature |
| Benchmark harness | Must | The acceptance criteria are measurements, so the |
|   |   | measuring instrument is itself a deliverable |
| Quality parity evaluation | Must | A cost number without an accuracy number is not a |
|   |   | result |
| Circuit breaker failover | Should | Highest value per unit of effort among the remaining |
|   |   | items, and independently demonstrable |


| Cost aware model routing | Should | Extends the cost story beyond caching |
| --- | --- | --- |
| OpenTelemetry tracing | Should | Turns claimed behaviour into observed behaviour |
| Adversarial false hit suite | Should | Tests the failure mode that matters most for a semantic |
|   |   | cache |
| SSE streaming passthrough | Could | Required for real adoption, not required for any |
|   |   | acceptance criterion |
| Per tenant fair share limiting | Could | Natural extension once single key limiting works |

## 8. User Stories

## US-001: Drop in adoption with free repeats

As an application developer, I want to point my existing OpenAI client at the gateway and have identical repeated prompts served from cache, so that I get caching without changing my application code or paying twice for the same answer.

## Acceptance criteria

- The gateway accepts the standard chat completions request body and returns the standard response shape, so a strict client parses it without error.

- Fields added by the gateway are additive, and a client that does not know about them ignores them.

- A second identical request does not reach a provider and returns a cache status of exact_hit .

- The same prompt sent against a different model is treated as a separate entry.

## US-002: Reworded questions served from cache

As an application developer, I want a prompt that means the same thing as an earlier one served from cache, and I want to see how confident that match was, so that natural phrasing variation does not cost me money and I can judge whether to trust the match.

## Acceptance criteria

- A paraphrase scoring at or above the configured similarity threshold returns the cached answer.

- The similarity score is returned on the response and recorded on the trace span.

- A paraphrase below the threshold goes to the provider.

- A near duplicate that changes meaning, such as an added negation, a swapped entity or an altered number, does not hit.

## US-003: Predictable spend per caller

As a platform engineer, I want a token budget enforced per API key so that one caller cannot exhaust the provider quota that other teams depend on.

## Acceptance criteria

- Requests are admitted against a token cost estimated before the provider call, not a request count.

- A caller over budget receives 429 with a retry hint, whether or not the answer happens to be cached.

- The estimate is reconciled against reported provider usage after the response returns, and the difference is released.

- Concurrent requests against one nearly empty bucket cannot both be admitted.

## US-004: Survives a provider outage

As an application developer, I want the gateway to keep serving when a provider degrades, so that an incident at one vendor does not take my application down with it.

## Acceptance criteria

- Repeated failures from the primary provider trip the circuit breaker open.

- While the breaker is open, traffic is served by the secondary provider with no caller visible error.

- The breaker probes the primary after a backoff with jitter and closes when it recovers.

- If the vector store is unavailable the gateway continues serving from Tier 1 rather than failing the request.

## 9. Product States

A request moves through a fixed set of states. The state reached is recorded on the trace span for every request, which is what makes the cache decision auditable after the fact.


```
RECEIVED
│
RATE_LIMIT_CHECKED ═╪═ rejected ═▶ THROTTLED (429)
│
TIER1_LOOKUP ═╪═ hit ═▶ SERVED_EXACT
│ miss
TIER2_LOOKUP ═╪═ hit ═▶ SERVED_SEMANTIC
│ miss
PROVIDER_CALL ═╪═ ok ═▶ CACHED_AND_SERVED
│ failure
BREAKER_EVALUATED ═╪═ failover ═▶ PROVIDER_CALL (secondary)
│ exhausted
FAILED (502)
```

## 10. Product Principles

- A measurement that cannot fail is not a measurement. Every headline claim has an experiment that could return an unfavourable number, and the unfavourable number gets reported.

- Budget enforcement is independent of cache state. A limit that behaves differently depending on what happens to be cached is not a limit anyone can reason about.

- A cache hit is a decision, not a lookup. Semantic hits are recorded with their similarity score so that a wrong answer can be traced back to the decision that produced it.

- Cheap checks run before expensive ones. This ordering is why the cache has two tiers rather than one.

- Degrade rather than fail. Losing the vector store should cost hit rate, not availability.

- 11. Scope

## 8.1 In scope for the MVP

- 1. FastAPI async gateway exposing an OpenAI compatible chat completions endpoint.

- 2. Tier 1 exact match cache keyed on a SHA-256 hash of the normalised prompt and model.

- 3. Tier 2 semantic cache using all-MiniLM-L6-v2 embeddings, run through the ONNX runtime, and ChromaDB similarity search, with a configurable threshold. See section 33.2.

- 4. Redis backed token bucket rate limiter keyed on estimated token counts from tiktoken, reconciled against reported provider usage after the call.

- 5. TTL and eviction policy across both cache tiers.

- 6. Benchmark harness that replays a fixed workload with caching on and off and reports cost, latency and hit ratio at three thresholds.

- 7. Quality parity evaluation on cache hits.

- 8. Docker Compose deployment covering gateway, Redis and ChromaDB with named volumes.

## 8.2 Stretch, in priority order

- 1. Circuit breaker failover across two providers, with closed, open and half open states and backoff with jitter.

- 2. Cost and latency aware routing between a cheap model and a frontier model.

- 3. OpenTelemetry tracing exported to a dashboard.

- 4. Adversarial false hit test suite covering negations, swapped entities and altered numbers.

- 5. SSE streaming passthrough.

- 6. Sharded vector index experiment measuring recall and latency against shard count.

- 7. Per tenant fair share rate limiting.

## 8.3 Out of scope

- Training or fine tuning any model, including the embedding model.

- A user interface. The deliverable is a service, and the benchmark report is its output.

- Multi region deployment, authentication beyond an API key, and billing integration.

- Guaranteeing that a semantic cache hit is always correct. The project measures the error rate, it does not eliminate it.

## 12. Technical Requirements

| ID | Requirement | Maps to |
| --- | --- | --- |
| TR-001 | Async request handling throughout, with no blocking calls on | BR-001, NFR throughput |
|   | the event loop |   |


| TR-002 | Deterministic prompt normalisation and hashing | BR-002 |
| --- | --- | --- |
| TR-003 | Embedding generation and vector search isolated behind one | BR-003 |
|   | interface so the model can be swapped |   |
| TR-004 | Atomic token bucket operations in Redis using a Lua script | BR-004 |
| TR-005 | TTL applied at write time in both tiers, with expiry enforced | BR-005 |
|   | on read |   |
| TR-006 | Benchmark harness runs headless, writes structured output, | BR-006 |
|   | and takes threshold as a parameter |   |
| TR-007 | Quality evaluation pipeline scoring cached answers against | BR-007 |
|   | fresh answers |   |
| TR-008 | Provider clients behind a common interface so a second | BR-008, BR-009 |
|   | provider is a configuration change |   |
| TR-009 | OpenTelemetry spans following the gen_ai semantic | BR-010 |
|   | conventions |   |
| TR-010 | Health endpoint reporting Redis and ChromaDB reachability NFR degradation |   |


## 13. High Level Design

The diagram below is the authoritative view of the system. Numbered blocks inside the gateway show the order in which a request is processed, which is the ordering the rest of this document defends.

*Figure 1. SemCache-Router high level architecture.*

## 13.1 Layer summary

## 13.2 Components

| Component | Responsibility |
| --- | --- |
| Gateway API layer | Request validation, OpenAI wire compatibility, response assembly |
| Rate limiter | Token estimation, bucket admission, post call reconciliation |
| Tier 1 cache | Hash lookup and write, TTL, hit accounting |
| Tier 2 cache | Embedding, vector search, threshold comparison, write on miss |
| Provider adapter | Uniform interface over OpenAI and Anthropic clients, usage extraction, cost computation |
| Circuit breaker | Failure accounting per provider, state transitions, backoff with jitter |


| Benchmark harness | Workload replay, metric collection, report generation |
| --- | --- |
| Evaluation pipeline | Quality scoring of cache hits against fresh generations |

## 14. Request Lifecycle

- 1. Request arrives and is validated against the chat completion schema.

- 2. The prompt is normalised and its token cost estimated with tiktoken.

- 3. The rate limiter attempts to reserve that estimate from the caller's bucket. If the reservation fails, the request is rejected with 429 and stops here.

- 4. Tier 1 is checked. On a hit the cached response is returned and the reservation is released, since no provider tokens were spent.

- 5. On a Tier 1 miss the prompt is embedded and Tier 2 is searched. If the nearest neighbour scores at or above the threshold, its response is returned and the similarity is recorded.

- 6. On a Tier 2 miss the router selects a model and the provider adapter issues the call, guarded by the circuit breaker.

- 7. On provider failure the breaker records it, and if the primary is open the request is retried against the secondary.

- 8. The response is written to both cache tiers with a TTL, the reservation is reconciled against reported usage, and the response is returned.

Ordering note. The rate limiter runs before the cache, not after. The alternative, serving cache hits for free and outside the budget, was considered and rejected. Free cache hits would make a caller's effective budget depend on the contents of a shared cache, which means the same caller sending the same traffic could be admitted or rejected depending on what an unrelated caller did a minute earlier. Budget enforcement that is unpredictable is not enforcement. The cost of this choice is that a caller at their limit is refused an answer that was already paid for, which is accepted deliberately.

## 15. Cache Design

13.3 Why two tiers

A single semantic tier would embed every incoming request before it could decide anything. Embedding is the most expensive step in the lookup path, both in latency and, if a hosted embedding API were used, in money. Literal repeats are common in real traffic and do not need semantic reasoning to be recognised. Tier 1 catches those in roughly a millisecond with a hash lookup, and only a Tier 1 miss pays for embedding and vector search.

## 13.4 Normalisation

The Tier 1 key is a SHA-256 hash over the model identifier and the normalised message list. Normalisation collapses runs of whitespace and trims each message, and does nothing else. In particular it does not lowercase and does not strip punctuation. Two prompts differing only in case are not reliably the same question, and folding them together at Tier 1 would hide a correctness risk inside what looks like a performance optimisation. That kind of near match belongs in Tier 2, where the threshold makes the trade off explicit and measurable.

## 13.5 Similarity and threshold

Tier 2 embeds the normalised prompt with a local all-MiniLM-L6-v2 model, run through the ONNX runtime that ChromaDB ships (section 33.2), and queries ChromaDB for the nearest stored prompt. A hit requires cosine similarity at or above the configured threshold. The collection is created in cosine space explicitly: ChromaDB defaults to squared L2 distance, under which one minus the distance is not a similarity (section 33.3). The benchmark exercises 0.75, 0.90 and 0.95, because the interesting behaviour of a semantic cache is not its performance at one threshold but the shape of the curve between hit rate and accuracy as the threshold moves.

The embedding model is local rather than a hosted API. Two reasons. First, a hosted embedding call would add real cost to every Tier 2 lookup, including lookups that miss, and that cost would have to be netted out of the headline cost reduction figure or it would silently inflate it. Keeping embeddings local keeps the cost ledger clean. Second, it removes a second external dependency from the request path, which matters for a project whose other headline claim is about surviving external failures.

## 13.6 TTL and eviction

Entries carry a TTL set at write time. On a hit the entry's hit count and last hit timestamp are updated but the remaining TTL is preserved rather than reset. TTL here represents freshness, not popularity, and a sliding expiry would let a frequently requested answer live indefinitely, which is the exact case where staleness is most visible.

## 16. Rate Limiter Design

The limiter is a token bucket held in Redis, keyed per API key and window. Capacity and refill rate are configuration. The unit is provider tokens rather than requests, because provider limits are expressed in tokens and a request count limit does not protect against a small number of very long prompts.

Because the true token cost of a response is not known until after the call, admission uses an estimate from tiktoken over the prompt plus the requested maximum completion length. The estimate is reserved at admission and reconciled once the provider reports actual usage, returning the difference to the bucket. Reservation and refill are performed in a


Lua script so that concurrent callers cannot interleave a read and a write and both be admitted.

## 17. Circuit Breaker Design

Each provider has a breaker with three states. Closed passes traffic and counts failures. When failures cross a threshold within a window the breaker moves to open, which rejects immediately without attempting a call, so that a failing provider is not sent traffic it cannot serve. After a backoff the breaker moves to half open and allows a limited number of probe requests. A success closes it; a failure returns it to open with a longer backoff.

The backoff uses exponential growth with jitter rather than plain exponential backoff. Without jitter, every client that failed at the same moment retries at the same moment, and the recovering provider receives a synchronised burst precisely when it is least able to absorb it. Jitter spreads those retries across the interval, which converts a thundering herd into a gradual ramp.

## 18. Database and Data Design

State lives in two stores. Redis holds the rate limiter buckets and the Tier 1 exact match cache, because both need sub millisecond key lookups and native TTL. ChromaDB holds the Tier 2 semantic index, because it needs vector similarity search rather than key equality. Nothing is stored in a relational database: there are no multi entity transactions, no joins, and no reporting queries against live state, so the operational complexity would buy nothing.

## 18.1 Entity relationships

```
│ API_KEY │ the caller identity
│ key_id (PK) │ and the budget boundary
│ capacity │
│ refill_rate │
│ 1
│
│ owns
│
│ N
│ RATE_LIMIT_BUCKET │ Redis
│ bucket_key (PK) │ rl:{api_key}:{window}
│ tokens_remaining │
│ reserved_estimate │
│ last_refill_at │
│ CACHE_ENTRY │ one logical record,
│ entry_id (PK) │ written to both tiers
│ prompt_hash (UK) │
│ response_body │
│ cost_usd │
│ expires_at │
│ 1 │ 1
│ │
indexed by indexed by
│ │
│ 1 │ 1
┌────▼─────┐ ┌────▼──────────┐
│ REDIS │ │ CHROMADB │
│ Tier 1 │ │ Tier 2 │
│ by hash │ │ by embedding │
│ TRACE_SPAN │ emitted per request,
│ trace_id (PK) │ exported, not stored
│ cache_status │
│ similarity_score │
│ breaker_state │
│ cost_usd │
```

A cache entry is one logical record with two physical representations. Tier 1 stores it in Redis under its prompt hash. Tier 2 stores the same response in ChromaDB indexed by the prompt embedding. Writes go to both on a provider miss, and the entry identifier is shared so that a hit in either tier is traceable to the same origin request.

## 18.2 Key design and indexes

| Key or index | Store | Purpose |
| --- | --- | --- |
| semcache:exact:v1:{hash} | Redis hash | Tier 1 primary lookup, TTL applied at write. A hash rather than a string; see section 33.1 |
| rl:{api_key}:{window} | Redis hash | Token bucket state, mutated only inside a Lua script |
| Embedding index | ChromaDB collection | Cosine similarity search for Tier 2 |


| entry_id | Both | Correlates a Tier 1 and Tier 2 record originating from the same |
| --- | --- | --- |
|   |   | response |

## 18.3 Data lifecycle

Write on provider miss, read on every request, expire on TTL, evict on capacity pressure. There is no archival tier: an expired cache entry has no value, because the whole point of expiry is that the answer may no longer be correct. Buckets are never deleted, only refilled, since a deleted bucket would silently grant a caller a full fresh allowance.

## 18.4 Field definitions

## Cache entry

| Field | Type | Notes |
| --- | --- | --- |
| entry_id | string | Primary identifier |
| prompt_hash | string | SHA-256 over model and normalised messages, Tier 1 key |
| embedding | float vector | Tier 2 index, present only in ChromaDB |
| prompt_text | string | Normalised prompt, retained for evaluation and debugging |
| response_body | object | Assistant message as returned |
| model_used | string | Model that produced the response |
| prompt_tokens | integer | Reported by provider |
| completion_tokens | integer | Reported by provider |
| cost_usd | float | Computed from reported usage and configured pricing |
| created_at | timestamp | Write time |
| expires_at | timestamp | Write time plus TTL |
| hit_count | integer | Incremented on each hit |
| last_hit_at | timestamp | Updated on each hit, does not extend TTL |

## Rate limiter bucket

| Field | Type | Notes |   |
| --- | --- | --- | --- |
| bucket_key | string | Pattern | rl:{api_key}:{window} |
| tokens_remaining | integer | Current balance |   |
| capacity | integer | Maximum balance |   |
| refill_rate | float | Tokens restored per second |   |
| last_refill_at | timestamp | Used for lazy refill on read |   |
| reserved_estimate | integer | Held against in flight requests, released on reconciliation |   |

## Trace span attributes

| Attribute | Value |
| --- | --- |
| gen_ai.system | Provider identifier |
| gen_ai.request.model | Model requested |
| gen_ai.usage.input_tokens | Prompt tokens |
| gen_ai.usage.output_tokens | Completion tokens |
| semcache.cache_status | exact_hit, semantic_hit or miss |
| semcache.similarity_score | Cosine similarity on a Tier 2 hit |
| semcache.breaker_state | closed, open or half_open |
| semcache.cost_usd | Cost of the request, zero on a hit |

## 19. API Specification

Base URL /v1 . The contract is deliberately the OpenAI contract: the gateway earns nothing by inventing its own request shape, and adopting it means an existing client changes one line.


## 19.1 Endpoint summary

| Method | Path | Auth | Purpose |
| --- | --- | --- | --- |
| POST | /v1/chat/completions | API key | Primary endpoint. Cached, rate limited, routed, failover protected. |
| GET | /health | None | Liveness plus dependency reachability |
| GET | /metrics | None | Aggregate counters consumed by the benchmark report |
| GET | /v1/cache/stats | API key | Hit ratio by tier, entry count, eviction count |
| DELETE | /v1/cache | API key | Flush both tiers. Used to force a cold start between benchmark |
|   |   |   | runs. |

## 19.2 POST /v1/chat/completions

## Request

```
POST /v1/chat/completions
Authorization: Bearer <caller_api_key>
Content-Type: application/json
{
"model": "gpt-4o-mini",
"messages": [
{ "role": "user", "content": "What is the capital of France?" }
],
"temperature": 1.0,
"max_tokens": 256
}
```

## Response, cache miss

```
HTTP/1.1 200 OK
{
"id": "chatcmpl-9f2a1c4e77b840d2ae13",
"object": "chat.completion",
"created": 1756531200,
"model": "gpt-4o-mini",
"choices": [{
"index": 0,
"message": { "role": "assistant", "content": "Paris." },
"finish_reason": "stop"
}],
"usage": {
"prompt_tokens": 14,
"completion_tokens": 3,
"total_tokens": 17
},
"semcache_status": "miss",
"semcache_similarity": null
}
```

## Response, semantic hit on a paraphrase

```
{
"model": "gpt-4o-mini",
"choices": [{ "message": { "role": "assistant", "content": "Paris." } }],
"usage": { "prompt_tokens": 14, "completion_tokens": 3, "total_tokens": 17 },
"semcache_status": "semantic_hit",
"semcache_similarity": 0.94
}
```

The two extra fields are additive. A strict OpenAI client ignores unknown keys, so compatibility is preserved while a client that wants to inspect cache behaviour can.

## 19.3 Request field reference

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| model | string | Yes | Part of the cache key. The same prompt against a different model is a separate |
|   |   |   | entry. |
| messages | array | Yes | Normalised before hashing and embedding |
| temperature | float | No | Defaults to 1.0. Not part of the cache key, which is a known simplification |
|   |   |   | recorded in section 32. |
| max_tokens | integer | No | Feeds the admission estimate, since the limiter must guess the completion cost |
|   |   |   | before it exists |
| stream | boolean | No | Accepted in the schema, honoured once SSE passthrough lands |


## 19.4 Error responses

```
{
"error": {
"code": "TOKEN_BUDGET_EXCEEDED",
"message": "Token budget exhausted for this key.",
"retry_after_seconds": 18,
"request_id": "req_7c31f9"
}
}
```

| Status | Code | Meaning |
| --- | --- | --- |
| 400 | INVALID_REQUEST | Body failed schema validation |
| 401 | MISSING_API_KEY | No caller key supplied |
| 429 | TOKEN_BUDGET_EXCEEDED | Bucket exhausted, carries a retry hint derived from the refill rate |
| 502 | ALL_PROVIDERS_UNAVAILABLE | Every configured provider failed or has an open breaker |
| 503 | DEPENDENCY_UNAVAILABLE | Redis unreachable, so admission cannot be decided and the gateway fails |
|   |   | closed |

## 19.5 Versioning

The path carries the version. A breaking change to the response shape means /v2 , not a silent alteration of /v1 , because callers are unmodified OpenAI clients that cannot be expected to adapt.


## 20. Low Level Design

This section drops from layers to modules. It shows the package layout, the interface of each core class, the sequence of a request that misses cache and hits a failing provider, the cache decision path, and the breaker state machine.

*Figure 2. Module structure, class interfaces, request sequence, cache decision path and breaker states.*

## 20.1 Design patterns used

| Pattern | Where |   | Why it is justified here |
| --- | --- | --- | --- |
| Strategy | LLMProvider | implementations | Adding Anthropic as a failover target must not require touching the |
|   |   |   | request path. Without this the breaker would be coupled to one |
|   |   |   | vendor SDK. |
| Adapter | Provider clients |   | OpenAI and Anthropic report usage in different shapes. The adapter |
|   |   |   | normalises them so cost accounting has one code path. |
| Dependency injection | Constructed in | lifespan , attached | Lets the test suite substitute a fake Redis and a stub provider without |
|   | to app state |   | patching module globals. |
| Repository | ExactCache , | SemanticCache | The route never sees Redis or Chroma directly, so a store can be |
|   |   |   | swapped without rewriting request handling. |

No pattern is used without a concrete need. There is no factory, no observer and no service locator, because nothing in the current scope requires them and each would add indirection a reviewer would have to read past.

## 20.2 Critical implementation details

- The embedding call runs in a thread pool. It is CPU bound. Awaiting it directly on the event loop would stall every other in flight request for the duration of a Tier 2 lookup, which would show up as a latency regression under concurrency and would be easy to misdiagnose as slow vector search.

- Bucket refill and reservation are one Lua script. Read, refill, compare and write execute atomically inside Redis. Doing this in Python would leave a window in which two concurrent callers both read a sufficient balance and both proceed.

- Reservations are released, not just reconciled. A cache hit spends no provider tokens, so the reservation is returned in full. A miss reconciles the estimate against reported usage and returns the difference.

- Failure classification is deliberate. Timeouts, 5xx and provider 429s count toward the breaker. A 400 does not, because a malformed request is a caller defect and would otherwise let one bad client trip failover for everybody.


- Cache writes are last write wins. Safe by construction here, since two answers to the same prompt are interchangeable, which is the same assumption the cache itself rests on.

## 21. Evaluation Methodology and Acceptance Criteria

## 21.1 Workload

The reference workload is the public SemBenchmarkLmArena dataset, which contains real chatbot prompts drawn from Chatbot Arena logs together with paraphrased variants. It is chosen for two reasons. It contains genuine paraphrase pairs, which is exactly the traffic pattern a semantic cache is meant to exploit, and it is the same dataset used in the published semantic caching benchmark this project takes as its reference point, so the comparison is like for like rather than a comparison against a workload constructed to flatter the result.

A fixed subset is sampled once, committed to the repository, and replayed identically across every configuration. Runs are started against a cold cache unless a run is specifically testing warm behaviour.

## 21.2 Quality parity method

Cached answers are scored against freshly generated answers for the same query using RAGAS answer relevancy and faithfulness. This was chosen over an LLM as judge equivalence check. A judge model introduces nondeterminism into the one metric whose credibility the project depends on, adds a cost that would have to be excluded from the cost figures, and invites the reasonable objection that the judging prompt was tuned until it agreed with the desired conclusion. RAGAS gives a repeatable score, which is what a falsifiable claim needs. Where RAGAS cannot express a comparison, an LLM judge is used for that narrow case only and reported separately.

## 21.3 The three acceptance demonstrations

These are the acceptance criteria for the project, not optional additions to it.

| No | Demonstration | Method | Target |
| --- | --- | --- | --- |
| 1 | Cost and latency benchmark | Replay the fixed workload with caching off, | Cost reduction of 40 percent or more at a |
|   |   | then on at thresholds 0.75, 0.90 and 0.95. | threshold whose accuracy is reported |
|   |   | Report cost, p50 and p95 latency, and hit ratio | alongside it |
|   |   | for each. |   |
| 2 | Quality parity guard | Score cache hit answers against fresh answers | Accuracy of 90 percent or above at |
|   |   | at each threshold. Plot accuracy against hit | whichever threshold produces the |
|   |   | rate. | headline cost figure |
| 3 | Chaos and failover | Fail the primary provider during a live run. | Breaker trips, traffic moves to the |
|   |   | Observe breaker transitions and caller visible | secondary, zero caller visible errors |
|   |   | outcomes. |   |

On the reference figures. The published ElastiCache semantic caching benchmark reported up to 86 percent cost reduction and up to 88 percent latency reduction at a 0.75 threshold while maintaining 91 percent accuracy, measured on 63,796 queries from SemBenchmarkLmArena using Claude 3 Haiku and Titan Text Embeddings V2. That result is cited here as the shape this project attempts to reproduce on its own workload and its own stack. Those numbers are not claimed as results of this project and will not be presented as such.

## 22. Success Metrics

| Metric | Definition |
| --- | --- |
| Cost reduction | Spend with caching on divided by spend with caching off, on the same workload |
| Cache hit ratio | Hits divided by total requests, reported separately for Tier 1 and Tier 2 |
| Latency p50 and p95 | End to end gateway latency, split by cache status |
| Answer accuracy | RAGAS score of cached answers against fresh answers, per threshold |
| False hit rate | Proportion of near duplicate prompts that should have missed but were served from |
|   | cache |
| Caller visible error rate during failover | Errors returned to the caller while the primary provider is failed |
| Rate limiter accuracy | Difference between estimated and actual token consumption after reconciliation |

## 23. Risks

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| Semantic cache returns a | High | High | This is the central risk and is measured rather than assumed away. The |
| plausible but wrong answer |   |   | adversarial suite specifically targets negations, swapped entities and |
|   |   |   | altered numbers, which are the cases where embeddings are closest |


|   |   |   | and meanings are furthest apart. |
| --- | --- | --- | --- |
| Threshold tuned until the | Medium | High | All three thresholds are reported, including unfavourable ones. The |
| numbers look good |   |   | headline figure always carries its accuracy. |
| Provider cost during | Medium | Medium | Sampled subset rather than the full 63,796 queries. Cached provider |
| benchmarking |   |   | responses are reused across configurations where the comparison |
|   |   |   | allows it. |
| Estimated tokens diverge from | Medium | Medium | Reconciliation after every call, and the divergence itself is reported as a |
| actual |   |   | metric rather than hidden. |
| Scope creep into stretch features | High | Medium | Stretch work does not begin until all three acceptance demonstrations |
| before the MVP is complete |   |   | produce numbers. |
| Embedding model download | Low | Medium Weights are fetched at image build time, not at request time, so a |   |
| unavailable in a locked down |   |   | running container needs no outbound access for embeddings. |
| environment |   |   |   |

## 24. Testing Strategy

| ID | Scenario | Expected result |
| --- | --- | --- |
| TEST-001 | Identical prompt sent twice | Second returns exact_hit, no provider call |
| TEST-002 | Same prompt, different model | Treated as a separate entry, provider is called |
| TEST-003 | Prompt differing only in surrounding whitespace | Same hash, exact_hit |
| TEST-004 | Paraphrase above threshold | semantic_hit with similarity recorded |
| TEST-005 | Paraphrase below threshold | Miss, provider is called |
| TEST-006 | Negated prompt, for example adding the word not Must miss, this is a false hit if it does not. Does not hold against the real model; see section 33.4 |   |
| TEST-007 | Entity swapped prompt, same structure different | Must miss |
|   | subject |   |
| TEST-008 | Numeric value changed in an otherwise identical | Must miss |
|   | prompt |   |
| TEST-009 | Entry read after TTL expiry | Miss, entry removed |
| TEST-010 | Caller exceeds token budget | 429 returned, cached or not |
| TEST-011 | Concurrent requests against one nearly empty | Admission is atomic, no overspend |
|   | bucket |   |
| TEST-012 | Actual usage lower than estimate | Difference returned to the bucket |
| TEST-013 | Primary provider fails repeatedly | Breaker opens, traffic moves to secondary |
| TEST-014 | Primary recovers while breaker is open | Half open probe succeeds, breaker closes |
| TEST-015 | ChromaDB unreachable | Tier 1 continues serving, degradation recorded, request does not |
|   |   | fail |
| TEST-016 | Redis unreachable | 503 returned, since admission cannot be decided |
| TEST-017 | Containers restarted mid workload | Cache and bucket state survive, run continues warm |
| TEST-018 | Load test at 50 concurrent callers | No queue growth, p95 within target |

Unit and integration tests run under pytest with a fake Redis, so the suite needs no running infrastructure. Load testing uses Locust. Tests numbered 006 through 008 form the adversarial false hit suite and are expected to be the hardest to pass, since they are the cases where a semantic cache is designed to be wrong.

## 25. Observability

Every request emits a span carrying the attributes listed in section 18.3. The reason for tracing rather than logging alone is that the interesting questions about this system are per request and comparative: which requests hit, at what similarity, at what cost, and in which breaker state. A log line answers that for one request. A trace answers it for a distribution.

Logged fields cover request identifier, cache status, similarity where applicable, provider and model, token counts, cost, breaker state and error code. Prompt text is logged only in a development configuration, since prompts are caller data.

Counters exposed on the metrics endpoint feed the benchmark report directly, so the harness reads the same numbers the service reports rather than computing its own parallel version.

26. Security Design


- Caller authentication is by API key, and the key is the rate limiter bucket identity, so authentication and budget are the same boundary.

- Provider credentials are read from the environment. They are never written into cache entries, spans, logs or the benchmark report.

- Cached responses are keyed by model and prompt only, and the cache is shared across callers. This is a deliberate trade off and a real one: a shared cache means one caller can receive an answer generated for another. It is acceptable for this project because the workload is a public dataset with no private content. Any deployment handling real user data would need the cache key namespaced per tenant, and that is recorded here rather than left as an implicit assumption.

- Request bodies are size limited, and prompt text entering the vector store is treated as data only.

- Compose exposes only the gateway port. Redis and ChromaDB are reachable on the internal network only.

## 27. Deployment Architecture

The stack runs under Docker Compose with three services: the gateway, Redis and ChromaDB. Redis and ChromaDB each mount a named volume, because state must survive a container restart. Without that, a benchmark run started immediately after a restart begins with a cold cache and reports a hit ratio that says nothing about the cache and everything about the restart. Redis runs with append only persistence rather than relying on periodic snapshots, since losing the last few seconds of writes to an untimely crash would quietly change the numbers between runs.

Environments are local development, then a single host deployment for the demo. Configuration is entirely by environment variables, so the same image runs in both. The embedding model is baked into the image at build time so that a running container does not need outbound network access to serve a request.

## 28. Cost Analysis

Cost is computed per request from reported token usage and a configured price table, rather than a fixed constant, so that a change in provider pricing does not silently invalidate every recorded result. The price table is committed alongside the benchmark output.

```
request_cost = (prompt_tokens * input_price_per_token)
+ (completion_tokens * output_price_per_token)
workload_cost = sum of request_cost over all cache misses
(hits contribute zero provider cost)
```

The comparison that matters is workload cost with caching off against workload cost with caching on, over the identical replayed workload. Embedding cost is zero in the ledger because embeddings are computed locally, which is one of the reasons that choice was made. Compute and storage for the three containers are not included, since the demonstration runs on a single host and those costs do not vary between the two configurations being compared.

## 29. 12 Week Project Roadmap

Twelve working weeks run from the start of September to the demo deadline of 25 November. The external viva follows in December. Each phase below names the artefact that proves it happened, because a phase with no artefact cannot be evaluated.

| Week | Phase | Deliverable | Evidence |
| --- | --- | --- | --- |
| 1 | Foundation | Repository scaffold, Compose stack, CI | GitHub submission, 6 September |
|   |   | pipeline, documentation set |   |
| 2 | Gateway core | OpenAI compatible endpoint, schema | Passing test suite |
|   |   | validation, Tier 1 exact cache |   |
| 3 | Provider integration | OpenAI adapter, tiktoken counting, cost | End to end request through the gateway |
|   |   | computation, cache write on miss |   |
| 4 | Checkpoint | Working miss path and Tier 1 hits demonstrable | Viva 1, 15 to 22 September |
|   |   | live |   |
| 5 | Semantic cache | Embedder, ChromaDB integration, similarity | Paraphrase served from cache |
|   |   | search, configurable threshold |   |
| 6 | Semantic cache | TTL and eviction across both tiers, degradation | Tests 004 to 009, 015 |
|   |   | when Chroma is down |   |
| 7 | Rate limiter | Token bucket, Lua atomicity, reservation and | Tests 010 to 012 |
|   |   | reconciliation |   |
| 8 | Benchmark | Harness, workload loader, first numbers at | Viva 2, 15 to 22 October |
|   |   | three thresholds |   |


| 9 | Evaluation | RAGAS quality parity pipeline, accuracy against | Acceptance demonstration 2 |
| --- | --- | --- | --- |
|   |   | hit rate curve |   |
| 10 | Resilience | Circuit breaker, Anthropic adapter, failover, | Acceptance demonstration 3 |
|   |   | adversarial false hit suite |   |
| 11 | Observability | OpenTelemetry spans, metrics endpoint, model | Viva 3, 15 to 22 November |
|   |   | routing |   |
| 12 | Delivery | Final benchmark report, deployment, demo | Demo video and deployment link, 25 November |
|   |   | recording, documentation |   |

The critical path runs through weeks 5 and 6. The semantic cache carries the most unknowns and gates two of the three acceptance demonstrations, so it is scheduled to complete before Viva 2 specifically to leave a month of recovery time if it does not work first time. Weeks 9 through 11 have deliberate slack; they will be consumed by whichever earlier phase overruns, and stretch features are drawn from section 11.2 only if they are not.

## 29.1 Team responsibilities

| Area | Student 1 |   |   | Student 2 |   |   |
| --- | --- | --- | --- | --- | --- | --- |
| Primary ownership | Gateway, caching, storage |   |   | Providers, evaluation, observability |   |   |
| Modules | api/ , | cache/ , | limiter/ | providers/ , | telemetry/ , | evaluation/ |
| Deliverables | Tier 1 and Tier 2 cache, token bucket, TTL and |   |   | Provider adapters, circuit breaker, benchmark |   |   |
|   |   |   | eviction, Compose stack | harness, RAGAS pipeline |   |   |
| Acceptance demonstration | Demonstration 1, cost and latency |   |   | Demonstrations 2 and 3, quality parity and |   |   |
|   |   |   |   | failover |   |   |
| Shared | Architecture decisions, code review on every pull request, testing strategy, CI, deployment, |   |   |   |   |   |
|   | documentation and the final presentation. Both members must be able to defend any decision in |   |   |   |   |   |
|   | section 30, not only the ones in their own modules. |   |   |   |   |   |

## 29.2 Definition of done

A feature is complete when it is implemented, covered by tests including its error paths, documented, integrated with the rest of the stack, and demonstrable in the deployed environment. A feature that works only when invoked directly from a test is not done.

## 29.3 Repository

All work is tracked at github.com/Anirudh0465. Branching is main for stable, feature/* for work in progress. Every pull request states the problem, the change, how it was tested and the risk it carries. Lint, tests and a secret scan gate every merge.

30. Architecture Decision Records

## ADR-001: Rate limiter runs before the cache

Decision. Admission control is evaluated before any cache lookup.

Rationale. Budget enforcement should not depend on cache contents. If cache hits were served free and outside the budget, a caller's effective allowance would vary with what unrelated callers had recently cached.

Trade off. A caller at their limit is refused an answer that costs nothing to serve. Accepted, because predictability is worth more here than the marginal hit.

## ADR-002: Two cache tiers rather than one

Decision. An exact hash tier is checked before the semantic tier.

Rationale. Embedding is the expensive step in the lookup path. Literal repeats do not need it, and they are common.

Trade off. Two stores to keep consistent and two TTL policies to reason about.

ADR-003: Local embedding model rather than a hosted embedding API

Decision. Embeddings are produced by a local all-MiniLM-L6-v2 model, run through the ONNX runtime rather than sentence-transformers (amended; see section 33.2).

Rationale. A hosted embedding call would add cost to every Tier 2 lookup including misses, contaminating the cost reduction figure, and would add a second external dependency to the request path.

Trade off. Model weights must be present in the image, and embedding quality is bounded by what a small local model can do.

## ADR-004: Case sensitive normalisation at Tier 1

Decision. Normalisation folds whitespace only. Case and punctuation are preserved.


Rationale. Case differences are not reliably meaning preserving. Folding them at Tier 1 hides a correctness decision inside a performance optimisation, where it cannot be measured.

Trade off. Lower Tier 1 hit rate. Those cases fall through to Tier 2 where the threshold makes the risk explicit.

## ADR-005: TTL preserved on hit rather than reset

Decision. A cache hit updates hit metadata but leaves the remaining TTL unchanged.

Rationale. TTL represents freshness. A sliding expiry would let popular entries live indefinitely, which is where stale answers do the most damage.

Trade off. Hot entries are regenerated on a fixed schedule even when demand is steady.

ADR-006: Backoff with jitter in the circuit breaker

Decision. Retry backoff is exponential with jitter.

Rationale. Synchronised retries from concurrent callers arrive as a burst exactly when a recovering provider is least able to absorb them.

Trade off. Recovery is slightly slower in the single client case, which is not the case worth optimising.

## ADR-007: RAGAS for quality parity rather than LLM as judge

Decision. Cache hit quality is scored with RAGAS answer relevancy and faithfulness.

Rationale. The quality figure is the credibility of the whole project. A judge model would make it nondeterministic, add cost that must be excluded from the cost figures, and open the objection that the judging prompt was tuned to agree.

Trade off. RAGAS may not capture every notion of equivalence. Where it cannot, a judge is used for that narrow case and reported separately.

## ADR-008: Named Docker volumes for Redis and ChromaDB

Decision. Both stateful services persist to named volumes, and Redis uses append only persistence.

Rationale. A benchmark that silently begins cold after a restart reports a number about the restart, not about the cache.

Trade off. Slightly higher write cost, and state must be explicitly reset between cold start runs.

*31. Traceability Matrix*

| Business | Technical | Component | Test | Demonstration |
| --- | --- | --- | --- | --- |
| BR-001 | TR-001 | Gateway API layer | TEST-001 | All |
| BR-002 | TR-002 | Tier 1 cache | TEST-001 to 003 | 1 |
| BR-003 | TR-003 | Tier 2 cache | TEST-004 to 008 | 1 and 2 |
| BR-004 | TR-004 | Rate limiter | TEST-010 to 012 | Not applicable |
| BR-005 | TR-005 | Both cache tiers | TEST-009 | 1 |
| BR-006 | TR-006 | Benchmark harness | TEST-018 | 1 |
| BR-007 | TR-007 | Evaluation pipeline | TEST-006 to 008 | 2 |
| BR-008 | TR-008 | Circuit breaker, provider adapter TEST-013, 014 |   | 3 |
| BR-009 | TR-008 | Router | TEST-013 | 1 |
| BR-010 | TR-009 | Tracing layer | TEST-015, 016 | 3 |

## 32. Future Scope

- Per tenant cache namespacing, which is the prerequisite for any deployment handling non public data.

- Adaptive thresholds learned per cached entry rather than one global constant, following the approach in the vCache line of work.

- A reranking pass over the top few vector candidates before accepting a hit, to reduce false hits without lowering the threshold globally.

- A guard on negation and changed numbers before a semantic hit is accepted. Section 33.4 shows no threshold defends against either, which moves this from optional to necessary for any deployment that serves real answers.

- Sharded vector index with recall and latency measured against shard count.

- Write through invalidation, so that a known stale answer can be evicted before its TTL expires.

- Support for additional providers behind the existing adapter interface.

- Including sampling parameters such as temperature in the cache key. They are excluded today, which means two requests differing only in temperature share an entry. That is acceptable for the benchmark workload, where temperature is fixed, and would not be acceptable in general.

## 33. Implementation Amendments

This section records where the implementation departs from the design above, and what building it measured. Each entry gives the reason, so a departure can be judged rather than discovered.

## 33.1 Tier 1 entries are a Redis hash, not a string

Section 18.2 specified a Redis string. A string holds the whole entry as one value, so recording a hit means reading it, editing the count and writing it back, and a plain SET clears the remaining TTL. That resurrects the entry for a full fresh lifetime on every hit, which is the sliding expiry section 13.6 rejects, arriving by accident with no error. SET with KEEPTTL avoids that but leaves the read and write racing, so two concurrent hits lose an increment. A hash fixes both: HINCRBY is atomic and HSET on an existing key does not touch its TTL, so preserving the TTL becomes a property of the data model rather than a rule every write has to remember. The key also carries a schema version, semcache:exact:v1, for the reason given in section 13.4.

## 33.2 Embeddings run through ONNX, not sentence-transformers

The model is unchanged: all-MiniLM-L6-v2, producing 384 wide unit vectors. Only the runtime differs. Resolving sentence-transformers on Python 3.14 pulls in torch and fifteen nvidia CUDA packages, none of which are needed to embed one sentence on a CPU. ChromaDB already depends on onnxruntime and tokenizers, which run the same model. The weights are a 79 MB download on first load, so they must be baked into the image at build time, and Tier 2 is off by default so that nothing outside the container downloads them unexpectedly.

## 33.3 Vector store behaviour that shaped the design

Three properties of ChromaDB 1.5.9 were verified rather than assumed, and each changed the implementation.

- Distance defaults to squared L2. For a pair with true cosine similarity 0.60, the default reported a distance from which one minus distance gives 0.20. The collection is created in cosine space, and a test asserts the reported similarity equals the constructed cosine, since a hit or miss check alone would not catch the default.
- Text without a vector is embedded by ChromaDB itself with its bundled model, and this happens even for a collection created with embedding_function set to None. Every call therefore passes a vector from the gateway's own embedder, and a test makes ChromaDB's embedder fail to prove it is never reached.
- The HTTP client raises at construction when the server is down. The client is therefore built lazily through a factory, so a vector store outage at startup degrades the gateway to Tier 1 rather than stopping it. After a failure the cache waits a configurable cooldown before reconnecting, so requests during an outage do not each pay for a connection attempt that will fail.

ChromaDB has no TTL. Expiry is stored as a timestamp in each entry's metadata and filtered in every query, so an expired entry is never served even before eviction removes it. The model is filtered the same way, so the same question asked of a different model never matches.

## 33.4 Measured findings

Cosine similarities from all-MiniLM-L6-v2 for pairs chosen to test the central claim of section 2:

| Pair | Relationship | Similarity |
| --- | --- | --- |
| What is your refund policy? / How do I get my money back? | Same meaning | 0.551 |
| What is the capital of France? / Tell me France's capital city. | Same meaning | 0.903 |
| How do I cancel my order? / How do I cancel my account? | Different meaning | 0.682 |
| What is 15 percent of 200? / What is 25 percent of 200? | Different meaning | 0.790 |
| Is the store open on Sunday? / Is the store not open on Sunday? | Opposite meaning | 0.972 |

At every threshold the benchmark sweeps, the paraphrase section 2 uses to motivate the project misses, and the negated question hits, serving the opposite answer. No single threshold separates them.

The adversarial probe in evaluation/adversarial.py generates negated, entity swapped and renumbered variants of a small sample and reports the false hit rate, the share that would be served when they should have missed:

| Mutation | 0.75 | 0.90 | 0.95 |
| --- | --- | --- | --- |
| Negation | 100% | 100% | 87.5% |
| Numeric change | 100% | 50% | 25% |
| Entity swap | 43% | 14% | 0% |
| All 19 cases | 79% | 58% | 42% |

Negated questions score between 0.96 and 0.98 against their originals. Raising the threshold trades hit rate for a lower entity swap error, but leaves negation almost undefended.

## 33.5 Consequences for the acceptance criteria

TEST-006 to TEST-008 state that these mutations must miss. Against the real model they do not, and asserting otherwise would fail for a property of the model rather than a defect in the code. They are therefore implemented as a measurement that reports the false hit rate, with one test that pins the negation finding so that a change in it is noticed. This is the reporting section 10 asks for: an experiment that can return an unfavourable number, with the unfavourable number reported.

The sample is small and hand written. Before these figures are quoted as general, the benchmark workload from section 21.1 should be run through the same probe.

## 33.6 Known limits

The model reads roughly the first 256 word pieces of its input, so a long conversation is effectively judged on how it opens. Two conversations that agree at the start and diverge later can score as near duplicates.
