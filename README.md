# SemCache-Router

A caching and reliability gateway that sits in front of LLM provider APIs like OpenAI and Anthropic.

## What this is

The gateway speaks the OpenAI chat completions wire format, so an application that already talks to OpenAI can adopt it by changing a base URL and nothing else. Behind that endpoint it adds the things a raw provider call doesn't give you: a two tier cache that catches both literal repeats and reworded ones, a rate limiter that counts tokens rather than requests, and failover to a second provider when the first one starts failing. Your client code doesn't change. It keeps sending the same requests, and the gateway decides whether any of them actually need to reach a provider.

Everything below describes the designed system. Nothing is implemented yet. See Project status for where things actually stand.

## Why this exists

Three things go wrong once an application built on a provider API starts seeing real traffic.

The first is paying twice for the same answer. People ask the same question in different words, and "what is your refund policy" and "how do I get my money back" usually want the same response. Exact string caching almost never catches this, because real users don't phrase things identically. So the provider bills you full token cost for a question you've already answered.

The second is that rate limits are measured in tokens, not requests. An application that carefully counts how many calls it makes will still get 429s, because a few long prompts can burn through a budget that hundreds of short ones wouldn't touch. Getting this right means estimating the cost of a request before you send it, not discovering it after the provider rejects you.

The third is having exactly one provider. When that provider has an incident, you're down too. Retrying straight away makes it worse: every client fails at the same instant, retries at the same instant, and the provider gets hit with a synchronised wave of traffic right when it's least able to handle it.

None of these are new problems. Caching, admission control and circuit breaking are ordinary distributed systems work. What's specific here is that a cache hit is decided by semantic similarity rather than key equality, which introduces something a normal cache never has to worry about: a hit can be wrong. Figuring out how often that happens, and at which threshold, is the real problem this project is about.

## How it works

A request moves through the gateway in this order.

1. The request arrives and is validated against the chat completion schema.
2. The prompt is normalised and its token cost is estimated with tiktoken.
3. The rate limiter tries to reserve that estimate from the caller's bucket. If there isn't room, the request is rejected with a 429 and stops here.
4. Tier 1, an exact hash lookup, is checked. On a hit the cached response goes straight back and the reservation is released, since no provider tokens were spent.
5. On a Tier 1 miss the prompt is embedded and Tier 2, the semantic cache, is searched. If the nearest stored prompt scores at or above the configured similarity threshold, its response is returned and the score is recorded.
6. If both tiers miss, the router picks a model and the provider adapter makes the call, guarded by a circuit breaker.
7. If the provider fails, the breaker records it, and traffic moves to the secondary provider.
8. The response is written to both cache tiers with a TTL, the reservation is reconciled against the usage the provider actually reported, and the response is returned.

Every request ends in one of a fixed set of states, and the state it reached is recorded on its trace span. That's what makes a cache decision auditable after the fact rather than something you have to take on faith.

```
RECEIVED
   |
RATE_LIMIT_CHECKED  -> rejected  -> THROTTLED (429)
   |
TIER1_LOOKUP        -> hit       -> SERVED_EXACT
   | miss
TIER2_LOOKUP        -> hit       -> SERVED_SEMANTIC
   | miss
PROVIDER_CALL       -> ok        -> CACHED_AND_SERVED
   | failure
BREAKER_EVALUATED   -> failover  -> PROVIDER_CALL (secondary)
   | exhausted
FAILED (502)
```

The rate limiter deliberately runs before the cache rather than after it. Serving cache hits for free sounds generous, but it makes a caller's budget depend on what's sitting in a cache they don't control, so the same caller sending the same traffic could be admitted or refused depending on what someone unrelated did a minute ago. A budget that unpredictable isn't really a budget. The price of this choice is that a caller at their limit gets refused an answer that was already paid for, and that's accepted on purpose.

## Cache design

Two tiers, because embedding is the expensive part of a lookup. A single semantic tier would have to embed every incoming request before it could decide anything at all, including requests that are word for word repeats of something answered a minute ago. Literal repeats are common in real traffic. Tier 1 catches them with a hash lookup in about a millisecond, and only a Tier 1 miss pays for embedding and vector search.

Tier 1 lives in Redis. The key is a SHA-256 hash over the model identifier and the normalised message list. Normalisation collapses runs of whitespace and trims each message, and does nothing else: it does not lowercase, and it does not strip punctuation. Two prompts that differ only in case aren't reliably the same question, and folding them together here would bury a correctness decision inside what looks like a performance tweak. That kind of near match belongs in Tier 2, where a threshold makes the trade off visible and measurable.

Tier 2 lives in ChromaDB. The normalised prompt is embedded with a local sentence-transformers model, the nearest stored prompt is retrieved, and a hit requires cosine similarity at or above the configured threshold. The benchmark runs at 0.75, 0.90 and 0.95, because the interesting thing about a semantic cache isn't how it performs at one threshold, it's the shape of the curve between hit rate and accuracy as you move it.

The embedding model is local rather than a hosted API, for two reasons. A hosted embedding call would add real cost to every Tier 2 lookup including the ones that miss, and that cost would have to be netted out of the headline cost figure or it would quietly inflate it. Keeping embeddings local keeps the ledger clean. It also removes a second external dependency from the request path, which matters for a project whose other main claim is about surviving external failures.

Entries carry a TTL set at write time. A hit updates the entry's hit count and last hit timestamp but leaves the remaining TTL alone. TTL here means freshness, not popularity. A sliding expiry would let a frequently requested answer live forever, and that's exactly the case where a stale answer does the most damage.

A cache entry is one logical record with two physical representations, sharing an entry identifier so a hit in either tier traces back to the same original response. On a provider miss, both tiers are written.

## Rate limiting

The limiter is a token bucket in Redis, keyed per API key and window. Capacity and refill rate are configuration, not constants in code. The unit is provider tokens rather than requests, because that's the unit providers actually enforce, and a request count limit does nothing to protect you from a handful of very long prompts.

The true cost of a response isn't known until the call comes back, so admission works on an estimate from tiktoken over the prompt plus the requested completion length. That estimate is reserved at admission and reconciled once the provider reports real usage, with the difference returned to the bucket. On a cache hit the reservation is released in full, since no provider tokens were spent.

Refill and reservation happen inside a single Lua script so the whole read, refill, compare and write sequence is atomic. Doing it in Python would leave a window where two concurrent callers both read a sufficient balance and both proceed, which is the exact bug a rate limiter exists to prevent.

## Failover and degradation

Each provider gets a circuit breaker with three states. Closed passes traffic and counts failures. Once failures cross a threshold inside a window, the breaker opens and rejects immediately without attempting a call, so a struggling provider stops receiving traffic it can't serve. After a backoff it moves to half open and allows a few probe requests through. One success closes it. One failure sends it back to open with a longer backoff.

The backoff is exponential with jitter. Without jitter, every client that failed at the same moment retries at the same moment, and a recovering provider gets a synchronised burst right when it's least able to absorb it. Jitter spreads those retries out, turning a thundering herd into a gradual ramp.

Failure classification is deliberate rather than generic. Timeouts, 5xx responses and provider 429s count toward the breaker. A 400 does not, because a malformed request is a defect in the caller, and counting it would let one badly behaved client trip failover for everyone else.

The vector store is treated as optional at request time. If ChromaDB is unreachable the gateway keeps serving from Tier 1 and records the degradation, rather than failing the request. Redis is different: if it's unreachable, admission can't be decided, so the gateway fails closed with a 503 rather than guessing.

## API

The contract is the OpenAI contract on purpose. Inventing a request shape would earn the project nothing and would cost every adopter a rewrite.

Endpoints:

- `POST /v1/chat/completions` requires an API key. The primary endpoint: cached, rate limited, routed and failover protected.
- `GET /health` is open. Liveness plus whether Redis and ChromaDB are reachable.
- `GET /metrics` is open. Aggregate counters, read directly by the benchmark report.
- `GET /v1/cache/stats` requires an API key. Hit ratio by tier, entry count, eviction count.
- `DELETE /v1/cache` requires an API key. Flushes both tiers, used to force a cold start between benchmark runs.

A request is an ordinary chat completions body:

```json
{
  "model": "gpt-4o-mini",
  "messages": [
    { "role": "user", "content": "What is the capital of France?" }
  ],
  "temperature": 1.0,
  "max_tokens": 256
}
```

The response is the ordinary shape with two fields added:

```json
{
  "model": "gpt-4o-mini",
  "choices": [{ "message": { "role": "assistant", "content": "Paris." } }],
  "usage": { "prompt_tokens": 14, "completion_tokens": 3, "total_tokens": 17 },
  "semcache_status": "semantic_hit",
  "semcache_similarity": 0.94
}
```

Both extra fields are additive. A strict OpenAI client ignores keys it doesn't recognise, so compatibility survives, while a client that wants to see what the cache did can look. Status is one of `exact_hit`, `semantic_hit` or `miss`, and similarity is null unless the hit came from Tier 2.

The model is part of the cache key, so the same prompt against a different model is a separate entry. `max_tokens` feeds the admission estimate, since the limiter has to guess the completion cost before the completion exists. Temperature is currently not part of the cache key, which is a known simplification and is listed under Known limitations.

Errors carry a code, a message and a request identifier:

- `400 INVALID_REQUEST` when the body fails schema validation.
- `401 MISSING_API_KEY` when no caller key is supplied.
- `429 TOKEN_BUDGET_EXCEEDED` when the bucket is exhausted, with a retry hint derived from the refill rate.
- `502 ALL_PROVIDERS_UNAVAILABLE` when every configured provider has failed or has an open breaker.
- `503 DEPENDENCY_UNAVAILABLE` when Redis is unreachable and admission can't be decided.

The version lives in the path. A breaking change to the response shape means a `/v2`, not a quiet change to `/v1`, because the callers are unmodified OpenAI clients and can't be expected to adapt.

## Configuration

Similarity threshold, TTL, bucket capacity and refill rate are all configurable without touching code, so a benchmark can sweep them. Configuration is entirely by environment variable, which means the same image runs locally and on the demo host. Provider credentials are read from the environment and never written into cache entries, logs, spans or the benchmark report. The embedding model weights are baked into the image at build time, so a running container needs no outbound network access just to embed a prompt.

## Tech stack

- Python: implementation language for the whole service.
- FastAPI: async gateway API, request validation, OpenAI wire compatibility.
- Redis: Tier 1 exact match cache and the rate limiter buckets, with atomic admission via a Lua script.
- ChromaDB: vector store for the Tier 2 semantic cache.
- sentence-transformers: local embedding model, so Tier 2 lookups add no API cost and no extra network dependency.
- tiktoken: token counting for cost estimation before a request is admitted.
- OpenAI and Anthropic APIs: the providers being fronted, one primary and one failover target.
- OpenTelemetry: per request spans carrying cache status, similarity, cost and breaker state.
- RAGAS: scoring cached answers against freshly generated ones for the quality parity check.
- pytest: unit and integration tests, run against a fake Redis so no infrastructure is needed.
- Locust: load testing at concurrency, checking the latency targets hold up.
- Docker Compose: runs the gateway, Redis and ChromaDB together so a benchmark run is reproducible.

## What makes this different from a wrapper

Anyone can forward a request to OpenAI. This project is built around three demonstrations instead of a feature list, and each one can fail.

The first is a cost and latency benchmark. The same fixed workload gets replayed with caching off, then on at 0.75, 0.90 and 0.95, reporting cost, p50 and p95 latency, and hit ratio for each. The goal is a meaningful cost reduction, reported next to the accuracy it came at rather than on its own.

The second is a quality parity check. Cached answers are scored against fresh answers for the same query using RAGAS answer relevancy and faithfulness, and accuracy is plotted against hit rate across thresholds. A cost number with no accuracy number beside it isn't a result, it's marketing. The aim is accuracy that holds up at whichever threshold produces the headline cost figure.

RAGAS was chosen over an LLM as judge on purpose. A judge model would put nondeterminism into the one metric the whole project's credibility rests on, add a cost that would then have to be excluded from the cost figures, and invite the entirely fair objection that the judging prompt got tuned until it agreed with the desired conclusion. Where RAGAS genuinely can't express a comparison, a judge is used for that narrow case and reported separately.

The third is a live chaos test. The primary provider is failed mid run, and the interesting question is what the caller sees while the breaker trips and traffic shifts to the secondary. The target is zero caller visible errors.

The workload is the public SemBenchmarkLmArena dataset, which is real chatbot prompts from Chatbot Arena logs together with paraphrased variants. It was picked because it contains genuine paraphrase pairs, which is precisely the traffic a semantic cache is supposed to exploit, and because it's the same dataset used by the published benchmark this project takes as its reference point. A fixed subset is sampled once, committed, and replayed identically across every configuration. Runs start cold unless the run is specifically about warm behaviour.

The PRD sets numeric targets for all three demonstrations, and cites a published benchmark as the shape it's trying to reproduce. Those are targets and references, not results. Nothing here has been measured yet, and when it has been, the numbers reported will be this project's own numbers on this project's own stack, including the unflattering ones.

## How it gets tested

Unit and integration tests run under pytest against a fake Redis, so the suite needs no running infrastructure. Load testing uses Locust.

The test list covers the obvious paths, an identical prompt served twice, the same prompt against a different model, whitespace differences, paraphrases above and below the threshold, TTL expiry, budget exhaustion, concurrent admission against a nearly empty bucket, reconciliation when actual usage comes in under the estimate, breaker transitions in both directions, both dependencies going away, a container restart mid workload, and 50 concurrent callers.

The three that matter most are the adversarial ones: a prompt with a negation added, a prompt with a swapped entity, and a prompt with a changed number. All three must miss. These are the cases where two prompts sit close together in embedding space while meaning something different, and they're expected to be the hardest tests to pass, because they target the exact failure a semantic cache is built to commit.

## Observability

Every request emits a span, following the OpenTelemetry `gen_ai` semantic conventions where they apply. Spans carry the provider and model, input and output token counts, cache status, similarity score on a Tier 2 hit, breaker state, and cost.

Tracing rather than logging alone, because the interesting questions here are per request and comparative: which requests hit, at what similarity, at what cost, in which breaker state. A log line answers that for one request. A trace answers it for a distribution. Prompt text is logged only in a development configuration, since prompts are caller data.

Counters on the metrics endpoint feed the benchmark report directly, so the harness reads the same numbers the service reports rather than computing a parallel version that could quietly disagree.

## Performance targets

- Tier 1 lookup under 5 ms at p95, Tier 2 lookup under 60 ms at p95 on the reference workload.
- 50 concurrent requests sustained on a single container with no queue growth, verified with Locust.
- Cache and limiter state survive a container restart, so a run started after a restart doesn't begin cold and report a number about the restart instead of about the cache.
- A benchmark run is reproducible from a committed workload file and a committed configuration. Reported numbers carry the git commit and the threshold they came from.

## Design decisions

The PRD records eight decisions with their trade offs. The short version:

- Admission control runs before the cache, so a budget doesn't depend on cache contents. Costs a caller at their limit an answer that was free to serve.
- Two cache tiers rather than one, because embedding is the expensive step and literal repeats don't need it. Costs two stores to keep consistent.
- A local embedding model rather than a hosted API, to keep the cost ledger clean and drop an external dependency from the request path. Costs image size and caps embedding quality at what a small local model can do.
- Whitespace only normalisation at Tier 1, preserving case and punctuation, so near matches are decided at Tier 2 where the threshold makes the risk explicit. Costs Tier 1 hit rate.
- TTL preserved on hit rather than reset, because TTL is about freshness and not popularity. Costs regeneration of hot entries on a fixed schedule.
- Backoff with jitter in the breaker, so concurrent callers don't retry in lockstep. Slightly slower recovery for a single client, which isn't the case worth optimising.
- RAGAS rather than an LLM judge for quality parity, for determinism and to avoid a tunable judging prompt. May not capture every notion of equivalence.
- Named Docker volumes with append only Redis persistence, so a restart doesn't silently reset a benchmark.

## Scope

In scope for the MVP: the OpenAI compatible endpoint, both cache tiers, TTL and eviction, the token bucket limiter, the benchmark harness, the quality evaluation, and the Compose deployment.

Stretch, in priority order: circuit breaker failover across two providers, cost aware routing between a cheap model and a frontier one, OpenTelemetry tracing to a dashboard, the adversarial false hit suite, SSE streaming passthrough, a sharded vector index experiment, and per tenant fair share limiting.

Explicitly out of scope: training or fine tuning anything including the embedding model, a user interface, multi region deployment, auth beyond an API key, and billing integration. Also out of scope is guaranteeing a semantic hit is always correct. The project measures that error rate. It does not claim to eliminate it.

## Known limitations

These are real and recorded rather than glossed over.

The cache is shared across callers and keyed on model and prompt only, which means one caller can receive an answer generated for another. That's acceptable here because the workload is a public dataset with nothing private in it. Any deployment touching real user data would need the cache key namespaced per tenant first.

Temperature isn't part of the cache key, so two requests differing only in temperature share an entry. Fine for a benchmark where temperature is fixed, not fine in general.

Cache writes are last write wins, which is safe by construction here only because the cache already assumes two answers to the same prompt are interchangeable.

Further out: adaptive per entry thresholds instead of one global constant, a reranking pass over the top few candidates to cut false hits without lowering the threshold everywhere, write through invalidation for answers known to be stale, a sharded index measured for recall against shard count, and more providers behind the existing adapter interface.

## Project status

Work in progress. The Tier 1 path works end to end against fakes: the same
prompt sent twice is served from cache, with the provider called once. Tier 2,
the rate limiter, the real provider adapters and the benchmark are not built.

- [x] Product requirements, architecture, API contract and design decisions written up
- [x] Repository scaffold: every module, test and config file in place, commented, no logic yet
- [ ] Compose stack filled in
- [x] OpenAI compatible endpoint and schema validation
- [x] Tier 1 exact match cache, with the TTL preserved on a hit
- [ ] Provider adapter, token counting and cost computation
- [ ] Tier 2 semantic cache, embedding and threshold search
- [ ] TTL and eviction across both tiers
- [ ] Token bucket rate limiter with reservation and reconciliation
- [ ] Benchmark harness and workload replay
- [ ] RAGAS quality parity pipeline
- [x] Circuit breaker
- [ ] Anthropic adapter and failover
- [ ] Adversarial false hit suite
- [ ] OpenTelemetry spans and metrics endpoint

The plan runs twelve weeks from early September to a demo on 25 November, with checkpoints in mid September, mid October and mid November. The critical path is the semantic cache, which carries the most unknowns and gates two of the three demonstrations, so it's scheduled early enough to leave recovery time if it doesn't work first try.

## Module tree

Every file below exists and carries a comment describing what will live in it. None of them contain logic yet.

```
app/
  __init__.py            package version
  main.py                FastAPI app, lifespan wiring, router registration
  config.py              environment driven settings, no constants elsewhere
  models.py              request, response and internal schemas
  api/
    chat.py              POST /v1/chat/completions, cache flush, cache stats
    health.py            GET /health, GET /metrics
  cache/
    keys.py              normalisation and SHA-256 key derivation
    exact_cache.py       Tier 1, exact match in Redis
    semantic_cache.py    Tier 2, vector similarity in ChromaDB
    embedder.py          local sentence-transformers encoder
  limiter/
    bucket.py            token bucket, reserve, reconcile, release
    scripts.lua          atomic refill and reserve
  providers/
    base.py              the provider interface
    openai_client.py     primary provider
    anthropic_client.py  failover target
    breaker.py           circuit breaker state machine
  telemetry/
    tracing.py           OpenTelemetry spans, gen_ai conventions
    metrics.py           counters behind /metrics
benchmarks/
  harness.py             workload replay across thresholds
  workload.py            committed workload subset loader
  report.py              report generation from the counters
evaluation/
  ragas_pipeline.py      quality parity scoring
  adversarial.py         negation, entity swap and numeric change generators
tests/
  conftest.py            fakeredis, stub provider, test client
  test_smoke.py          imports every module, keeps CI green from commit one
  test_cache.py test_limiter.py test_breaker.py
  test_api.py test_adversarial.py
locustfile.py            load profile for the concurrency target
docker-compose.yml       gateway, redis, chromadb
Dockerfile               bakes the embedding model at build time
```

## Quickstart

Copy the example environment and fill in your provider keys first. `.env` is git ignored and must never be committed.

```
cp .env.example .env
docker compose up
```

The gateway is the only service with a published port. Redis and ChromaDB stay on the internal network. Once it's up:

```
curl http://localhost:8000/health
```

### Running it now, without Docker

The gateway runs on the stub provider with no credentials and no network. Redis is the only thing it needs:

```
uv venv -p 3.13 .venv
uv pip sync requirements.lock.txt
.venv/bin/uvicorn app.main:app
```

Then, in another shell, send the same question twice with different spacing:

```
curl -s localhost:8000/v1/chat/completions -H 'Authorization: Bearer test' -H 'Content-Type: application/json' -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"What is the capital of France?"}]}'
curl -s localhost:8000/v1/chat/completions -H 'Authorization: Bearer test' -H 'Content-Type: application/json' -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"  What is   the capital   of France?  "}]}'
```

The first reports `"semcache_status": "miss"`, the second `"exact_hit"` with identical content. That pair proves normalisation, hashing and the Tier 1 cache in one go.

Tier 2, the token budget and provider failover are not wired yet, so `docker compose up` gives you the stack but not those behaviours.

## Environment variables

Every value is read from the environment. There are no constants in code, so a benchmark can sweep any of these without a code change.

| Variable | Default | What it controls |
| - | - | - |
| `OPENAI_API_KEY` | none | Primary provider credential. Required. Never committed. |
| `ANTHROPIC_API_KEY` | none | Failover provider credential. Required. Never committed. |
| `SEMCACHE_REDIS_URL` | `redis://redis:6379/0` | Tier 1 cache and rate limiter buckets. |
| `SEMCACHE_CHROMA_HOST` | `http://chromadb:8000` | Tier 2 vector index. |
| `SEMCACHE_SIMILARITY_THRESHOLD` | `0.90` | Theta. A Tier 2 hit needs a score at or above this. |
| `SEMCACHE_CACHE_TTL_SECONDS` | `3600` | Entry lifetime, applied at write time, never extended on a hit. |
| `SEMCACHE_BUCKET_CAPACITY` | `100000` | Maximum tokens a caller may hold. |
| `SEMCACHE_BUCKET_REFILL_RATE` | `1000` | Tokens restored per second. Also derives the retry hint on a 429. |
| `SEMCACHE_PRIMARY_PROVIDER` | `openai` | Provider tried first. |
| `SEMCACHE_SECONDARY_PROVIDER` | `anthropic` | Failover target. |
| `SEMCACHE_BREAKER_FAILURE_THRESHOLD` | `5` | Failures in the window before the breaker opens. |
| `SEMCACHE_BREAKER_WINDOW_SECONDS` | `60` | Window those failures are counted in. |
| `SEMCACHE_BREAKER_BASE_BACKOFF` | `1.0` | Starting retry delay in seconds. |
| `SEMCACHE_BREAKER_BACKOFF_CEILING` | `60.0` | Cap that exponential growth stops at. |
| `SEMCACHE_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Encoder baked into the image at build time. |
| `SEMCACHE_LOG_PROMPTS` | `false` | Whether prompt text reaches the logs. Development only, since prompts are caller data. |


## Repository structure

The module tree above is the layout, and it now matches the PRD's low level design file for file. Alongside it sit the PRD itself, this README, the Compose stack, the Dockerfile, the CI workflow and the example environment.

This is an OJT self assign project at Polaris School of Technology on the Generative AI track. The full design, including the traceability matrix and the complete test list, is in the [PRD](<SemCache-Router_PRD3 (1).md>).
