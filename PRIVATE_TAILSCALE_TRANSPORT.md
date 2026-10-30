# Private Tailscale ASR transport

## Scope

This image is a child of the accepted ASR image `myblockchaincompany/jupyter-salad:sha-80fbf5d`, pinned to manifest digest `sha256:1c5d4a04f46ac878439fa40133b324038e426eefa9537dba4fef5b8087e8df09`. Its only runtime additions are the pinned Tailscale 1.104.4 `tailscale` and `tailscaled` binaries and a wrapper that starts the daemon in userspace mode before running the original Salad entrypoint. The ASR image, nginx configuration, and original entrypoint are not changed.

The Tailscale amd64 archive is fetched from the official package host using Docker BuildKit's SHA-256 checked `ADD`. Its archive SHA-256 is `624ea093d47d62cf56f17b8332d74b8decda4a1faa610ebc60a1de334b2b3dc9`.

## Network path

The experiment uses the node's private Tailscale IPv4 and nginx's existing port 8888:

```text
Browser
  -> http://127.0.0.1:8765/asr/benchmark
  -> Windows TCP portproxy
  -> Tailscale private IPv4:8888
  -> userspace Tailscale TCP forwarding
  -> container localhost:8888 (existing nginx)
  -> existing ASR / Jupyter services
```

The public Container Gateway must not be used for the benchmark. The first private experiment keeps nginx so only the Gateway path changes.

Tailscale's userspace FAQ states that incoming connections are proxied to the same port on the node's `127.0.0.1`. This is the basis for testing `<TAILSCALE_IP>:8888` against the already-running local nginx listener without Tailscale Serve. The Salad image binds nginx on `0.0.0.0:8888`, and its existing `/asr/` location forwards to Uvicorn on `127.0.0.1:8765`. Sources: [Tailscale userspace networking](https://tailscale.com/docs/concepts/userspace-networking), [Tailscale userspace incoming-connection behavior](https://tailscale.com/docs/reference/faq/other-vpns).

Experiment path: `EXPERIMENT_PATH=TAILSCALE_PRIVATE`

## Salad container setup

Create a new single-replica test Container Group from the published `private-tailscale-<full-commit-sha>` image. Do not edit the existing group. Keep the Qwen model, runtime, environment, and ASR settings identical to the accepted live baseline, including `QWEN_MODEL_CHUNK_MS=1000` and `QWEN_LANGUAGE=es`. Supply `TAILSCALE_AUTH_KEY` using Salad's secret environment-value control. Use an ephemeral key; do not paste it into chat, a file, a build argument, a log, or an image. The wrapper runs `tailscaled --tun=userspace-networking --state=mem:`, requires a non-empty `SALAD_MACHINE_ID` and passes that exact value as the Tailscale hostname, and removes the auth key from its environment before starting the unchanged app entrypoint. It explicitly leaves Tailscale SSH disabled and does not enable Serve, MagicDNS, or a new ASR port. Tailscale documents memory-only state and ephemeral node handling [here](https://tailscale.com/docs/features/ephemeral-nodes); its auth-key guidance recommends ephemeral keys for short-lived containers [here](https://tailscale.com/docs/features/access-control/auth-keys).

For the controlled experiment, disable the Container Gateway if Salad allows the group to run without it. Otherwise, leave it only as an unused health/debug path and never use its public domain for the microphone session.

## Windows client route

With Tailscale active on Windows and the test node's private IPv4 shown in Salad container logs, verify the node using `tailscale ping <TAILSCALE_IP>`. Then from an elevated PowerShell prompt create:

```powershell
netsh interface portproxy add v4tov4 listenaddress=127.0.0.1 listenport=8765 connectaddress=<TAILSCALE_IP> connectport=8888
```

Confirm the forwarding entry with `netsh interface portproxy show v4tov4`, then open `http://127.0.0.1:8765/asr/benchmark`. The local address keeps microphone access in a secure browser context. The public Salad domain is not the experiment path.

Remove only this forwarding entry after the test:

```powershell
netsh interface portproxy delete v4tov4 listenaddress=127.0.0.1 listenport=8765
```

## Runtime checks and result gates

Before the user speaks, record the Container Group version, instance UUID, machine ID, GPU/driver, image tag and digest where Salad exposes them, Tailscale node name/private IP, runtime versions and hashes, health/readiness, model-loaded state, and private TCP reachability to port 8888. Keep the browser CloseEvent wrapper and capture the server's existing telemetry incrementally outside the ephemeral instance. Do not add ASR instrumentation for this first test.

Run one human-speech session with `model_chunk_ms=1000` and Spanish, targeting at least 10 minutes and preferably 15. Preserve the raw transcript and telemetry outside the Salad instance before it can be reallocated. Classify a stable session of at least 10 minutes as `CONTAINER_GATEWAY_BYPASS_STABLE` and `PUBLIC_GATEWAY_MATERIALLY_IMPLICATED`; this does not establish the Gateway's internal cause. Classify another `1006` on the private path as `PRIVATE_PATH_DROP_REPRODUCED`. Do not automatically start a direct-Uvicorn follow-up.

## Remote image verification

The manual GitHub Actions workflow builds the pinned child image on a remote Linux runner without publishing it first. Before pushing the unique commit-specific tag, it verifies that:

- the inherited Docker configuration is unchanged except for the expected wrapper `CMD`;
- the child image keeps the exact base rootfs layer sequence as its prefix and adds layers;
- SHA-256 inventories match for the existing nginx configs, ASR source tree, Qwen/vLLM package trees, and original entrypoint;
- both Tailscale binaries report version 1.104.4;
- the wrapper exits with its documented status when the secret is absent;
- the image history contains no Tailscale auth-key literal.

Only after these checks pass does the workflow authenticate to Docker Hub and publish `private-tailscale-<full-commit-sha>`. It does not publish a stable, latest, or base-image SHA tag. The image smoke check overrides the inherited entrypoint and does not start Jupyter, Uvicorn, nginx, Qwen, CUDA, or a model.

## Current execution status

`IMPLEMENTATION_PREPARED; SOURCE_CHECKS_PASS; REMOTE_BUILD_AND_IMAGE_SMOKE_PENDING; SALAD_DEPLOYMENT_PENDING; PRIVATE_CONNECTIVITY_PENDING; HUMAN_SPEECH_TEST_PENDING`.

The contract check is at repository root rather than under `tests/` because the existing base-image workflow triggers on every `tests/**` path and publishes stable tags. Variant-only changes must not trigger that workflow.

This status changes only with recorded CI, Salad, Tailscale, network, and browser evidence. Source checks do not establish image publication or a successful private runtime.
