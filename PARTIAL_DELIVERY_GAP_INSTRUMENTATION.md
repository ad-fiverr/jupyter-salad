# SCRUM-197 — Partial Delivery Gap Instrumentation

## Scope and gate

This is instrumentation-only for the next separately authorized human speech run. It does not change Qwen decoding, scheduler admission, rolling epochs, handoff, stitching, buffers, nginx, Tailscale, MTU, or the existing ASR request behavior. The overlay is not built, published, or deployed in this phase. No audio or GPU run is authorized by this document.

The accepted baseline remains `model_chunk_ms=1000`, unfixed `2/5`, the private Tailscale path, MTU 1100, and unchanged nginx. The prior export recorded effective language `auto`; the UI intention `es` is not evidence that an explicit `es` reached the runtime. The next export must remain authoritative for effective language.

## Correlation identity and clocks

Candidate rows use `partial_id = <public_stream_id>/<epoch_seq>/<revision>`. `qwen_local_stream_id` is retained separately because it changes across epochs. B0 rows for empty/unchanged text have no `partial_id`; their cursor and outcome identify a decode opportunity without inventing a candidate. Rows store UTF-8 byte length, stable FNV-1a 32-bit text fingerprint, language when present, UTC timestamp, and a monotonic timestamp. Diagnostic rows never contain transcript text, PCM, context, or API credentials. The 32-bit fingerprint is a content-comparison aid, not a security or authentication primitive.

UTC timestamps are suitable for coarse cross-process ordering. Monotonic timestamps are local to their process and must not be subtracted across server and browser clocks.

## Instrumented boundaries

| Boundary | Trace marker | Observation point |
| --- | --- | --- |
| B0 — decode push result | `decode_push_result` | Immediately after every successful `push` decode returns `reply.text`, including `decode_steps_delta=0`, before the existing empty/unchanged early return. Records cursor, exact decode-step delta, outcome, and previous/new revision. Only `changed_text_candidate` maps to a deterministic partial identity; empty/unchanged outcomes keep both partial identity fields null. A zero-step result proves a completed push response, not an inference step. |
| B1 — raw runtime candidate | `runtime_candidate` | Immediately before `QwenStreamingRuntime._filter_candidate_result`. The structured raw `partial_candidate` has been constructed, but no canonical fence, validation, or stitching filter has run. This is not a model-token hook. |
| B1-filter — canonical result | `canonical_filter` | Records the actual canonical filter result for the same partial identity: `accepted`, `stale_fence`, `invalid_scope`, `classify_error`, `invalid_candidate`, `stitch_rejected`, or `unbound_fence`. Rejections are observed at their existing rejection branch; acceptance is observed only after the outer result validation succeeds. |
| B2 — server event prepared | `server_event` | Immediately after the existing public event projection and before it enters the outbound event queue. |
| B3 — WebSocket send | `ws_send`, `phase=pre/post` | Under the existing send lock, immediately before and after `websocket.send_json`. Post outcome is `sent` or `error:<exception type>`. |
| B4 — browser message event | `browser_message_event` | Uses `MessageEvent.timeStamp`, captured by the WebSocket handler. The row is appended after JSON parsing so it can carry the partial identity, while retaining the event timestamp. |
| B5 — parsed `onmessage` | `browser_onmessage_parsed` | Timestamp immediately after `JSON.parse`, before dispatch to transcript handling. |
| B6 — dashboard append | `stream_events_append` | Immediately after the existing `state.streamEvents.push(item)` and before its existing 1,000-item retention trim. |
| B7 — JSON export | `export_json` | One metadata row per partial actually present in `streaming.partials` immediately before JSON serialization. |

B4 is the first JavaScript-visible message event timestamp; browsers do not expose an independent wire-arrival timestamp for each WebSocket frame. The B4 row is correlated after parse, using the event timestamp captured before entering the handler. A JSON parse failure therefore cannot produce an identity-bearing B4 row and remains `UNKNOWN` for that partial.

The client stores only bounded metadata rows in an 8,192-entry circular buffer. It does not write to browser storage, print per-partial console logs, or add renders. The server keeps a bounded 8,192-row in-memory trace containing B0 through B3. The browser trace contains B4 through B7. Both report dropped-row counts. The server trace endpoint is authenticated and read-only:

`GET /asr/partial-delivery-trace?stream_id=<public_stream_id>`

Use the existing ASR bearer token in the request header. Fetch and preserve this JSON before destroying or reallocating an ephemeral instance. The browser benchmark JSON contains B4–B7 trace rows under `streaming.partial_delivery_diagnostics.events`; its existing `streaming.partials` array remains the source of transcript text.

## Offline correlation

`asr-lab/benchmark/partial_delivery_correlate.py` consumes the saved server endpoint response and browser benchmark export. It reports:

- last observed partial identity and timing at every boundary;
- WebSocket send success/error counts;
- first missing downstream boundary for any observed upstream identity, even if later identities recover;
- text fingerprint/length/language mismatch for an identity that appears on both sides of a boundary;
- input SHA-256 values;
- whether bounded traces dropped rows.

If every observed delivery boundary tail agrees and neither bounded trace dropped rows, the result is `PARTIAL_GAP_NOT_REPRODUCED` only for the latest observed candidate identity. B0 records every successful push result, including zero-step replies; `DECODE_CONTINUES_NO_TEXT_CHANGE` requires an advancing cursor, nonzero decode steps, and `empty_text`/`unchanged_text`. A changed B0 without matching B1, a B1 without a filter outcome, or a missing/unrecognized filter outcome remains `UNKNOWN`; only an explicit known rejection outcome is classified as rejection. Absence alone is not converted into a causal claim. Any dropped rows or invalid completeness count forces `UNKNOWN`.

Diagnostic payloads are allowlisted to identity and content-free metadata. The sanitizer never copies `text`, `stitch_raw_current_text`, PCM, context, token data, or credentials, including from accepted stitched candidates.

The expected causal classifications for an observed identity are:

| Missing transition | Classification |
| --- | --- |
| decode cursor advances, decoded text is empty/unchanged | `DECODE_CONTINUES_NO_TEXT_CHANGE` |
| changed B0 result has no matching B1/filter evidence | `UNKNOWN` |
| B1 raw candidate → canonical filter rejection | `RAW_CANDIDATE_REJECTED` |
| canonical filter accepts candidate, but B2 is absent | `ACCEPTED_CANDIDATE_NO_B2` |
| server event → successful WebSocket send | `SERVER_WS_SEND_STOP` |
| successful send → browser message event | `FRAME_DELIVERY_STOP` |
| message event → handler entry | `BROWSER_ONMESSAGE_STOP` |
| handler entry → `streamEvents` append | `STREAM_EVENTS_APPEND_STOP` |
| append → export | `EXPORT_OMISSION` |

At export, the client marks a retained `state.streamEvents` partial that is absent from the exported partial array as an omission candidate; older rows already evicted by the existing 1,000-event retention are not treated as expected exports. No upstream identity or incomplete/truncated evidence produces `UNKNOWN`. A final candidate is not proof that every PCM sample was accounted for.

## Diagnostic overlay identity

`Dockerfile.partial-delivery-gap` pins the already published private-Tailscale base by both tag and digest:

`myblockchaincompany/jupyter-salad:private-tailscale-9125118440141e0e22796423b38c062565ccff7d@sha256:ea3df64f62cb5fdce643bb7a7fd175696f51ee04c9c61522f18d51aa783cb63b`

The overlay copies exactly five modified runtime files into their existing `/opt/asr-lab` locations: `qwen_streaming.py`, `asr_qwen_epoch_lifecycle.py`, `asr_service_lifecycle.py`, `service.py`, and `benchmark_web/app.mjs`. It adds no packages, entrypoint, port, startup behavior, or model files. Build, publication, deployment, and the human run require the later review gate and separate authorization.
