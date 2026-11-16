[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/bibitcuan1-boop/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/bibitcuan1-boop/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

# free-model-aggregator

The best live free-model aggregator: one Node 22 server (`main.ts`, standard
library only, no dependencies) that answers OpenAI- and Anthropic-compatible
requests with real completions from the best-rated free models on OpenRouter,
dialled through the grader's gateway.

Built for the [make.fun](https://make.fun/bounty/free-model-aggregator-02)
bounty **"The best live free-model aggregator"**
([free-model-aggregator-02](https://make.fun/bounty/free-model-aggregator-02)).

## What it does

- `POST /v1/chat/completions` — OpenAI-compatible, JSON and streaming SSE.
- `POST /v1/messages` — Anthropic-compatible, JSON and streaming SSE
  (translated to and from Chat Completions; every free model is dialled on the
  gateway's Chat Completions route).
- `GET /v1/models` — the OpenAI list shape for `Authorization: Bearer`, the
  Anthropic list shape for `x-api-key`; both list exactly the fourteen
  dialable free models.
- `node main.ts --list` — prints the catalog document, byte-for-byte the one
  it registers with the env.

## How it picks a model

Candidates are ranked by the brief's `rating_milli` among the models that meet
the request's requirements, so a JSON-mode prompt goes to the best
JSON-capable model (NVIDIA Nemotron 3.5 Super VL, 1303) and a plain or
tools prompt to the best model overall (Thinking Machines Inkling 2 Small,
1475). A model that refuses (429 / 5xx) is never retried in the same trial:
the next best candidate serves instead, and the refusal trail is reported on
the `x-makefun-upstream-failures` header so the failover column can see it.

## How it runs in a trial

The server listens on `0.0.0.0` at a free port, registers that port and its
catalog with `POST $MAKEFUN_ENV_URL/register`, answers the six prompts the env
sends it, polls `GET $MAKEFUN_ENV_URL/work`, and exits 0 when the work is
done. The OpenRouter key never enters the server: the gateway swaps the
trial's dummy key for the key pushed to the competition. Model choice is a
pure function of the brief and the gateway's answers — no wall clock, no
unseeded randomness — so a replay leaves the same state.

## Files

- `main.ts` — the whole server: brief parsing, catalog, both wires, routing,
  failover, register and poll.
- `LICENSE` — MIT.
