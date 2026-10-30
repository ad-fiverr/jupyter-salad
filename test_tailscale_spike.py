import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
dockerfile = (ROOT / "Dockerfile.tailscale-spike").read_text(encoding="utf-8")
entrypoint = (ROOT / "tailscale-salad-entrypoint.sh").read_text(encoding="utf-8")
workflow = (ROOT / ".github/workflows/tailscale-spike.yml").read_text(encoding="utf-8")
documentation = (ROOT / "PRIVATE_TAILSCALE_TRANSPORT.md").read_text(encoding="utf-8")

assert "myblockchaincompany/jupyter-salad:sha-80fbf5d@sha256:1c5d4a04f46ac878439fa40133b324038e426eefa9537dba4fef5b8087e8df09" in dockerfile
assert "tailscale_1.104.4_amd64.tgz" in dockerfile
assert "624ea093d47d62cf56f17b8332d74b8decda4a1faa610ebc60a1de334b2b3dc9" in dockerfile
assert "--tun=userspace-networking" in entrypoint
assert "--state=mem:" in entrypoint
assert "--auth-key=\"$TAILSCALE_AUTH_KEY\"" in entrypoint
assert "--ssh=false" in entrypoint
assert 'if [[ -z "${SALAD_MACHINE_ID:-}" ]]' in entrypoint
assert entrypoint.index('if [[ -z "${SALAD_MACHINE_ID:-}" ]]') < entrypoint.index("/usr/local/bin/tailscaled")
assert "SALAD_MACHINE_ID must be provided by Salad." in entrypoint
assert '--hostname="$SALAD_MACHINE_ID"' in entrypoint
assert "private-test" not in entrypoint
assert "salad-asr-" not in entrypoint
assert "EXPERIMENT_PATH=TAILSCALE_PRIVATE" in documentation
assert "unset TAILSCALE_AUTH_KEY" in entrypoint
assert "ENTRYPOINT" not in dockerfile
assert "salad-jupyter-entrypoint" in entrypoint
assert "tailscale serve" not in entrypoint
assert "load: true" in workflow
assert "push: false" in workflow
assert "private-tailscale-${{ github.sha }}" in workflow
assert "docker push \"$IMAGE_TAG\"" in workflow
assert "jupyter-salad:latest" not in workflow
assert "tags: myblockchaincompany/jupyter-salad:sha-" not in workflow
assert "secrets.TAILSCALE_AUTH_KEY" not in workflow
assert "TAILSCALE_AUTH_KEY=" not in workflow
assert "RootFS" in workflow and "sha256sum" in workflow and '"docker", "history"' in workflow
assert workflow.index("Verify child image delta and smoke test before publication") < workflow.index(
    "Log in to Docker Hub after validation"
) < workflow.index("Publish only the isolated commit-specific variant tag")

for source in (dockerfile, entrypoint, (ROOT / "PRIVATE_TAILSCALE_TRANSPORT.md").read_text(encoding="utf-8")):
    assert re.search(r"tskey-[A-Za-z0-9_-]{12,}", source) is None
assert "ARG TAILSCALE_AUTH_KEY" not in dockerfile
assert "ENV TAILSCALE_AUTH_KEY" not in dockerfile

print("Tailscale variant contract checks passed.")
