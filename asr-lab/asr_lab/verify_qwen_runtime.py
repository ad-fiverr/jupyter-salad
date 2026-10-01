"""Import-only contract for the isolated Qwen/vLLM environment; never loads weights."""
from __future__ import annotations

import importlib.metadata


def verify() -> dict[str, str]:
    expected = {
        "qwen-asr": "0.0.6",
        "vllm": "0.14.0",
    }
    observed = {name: importlib.metadata.version(name) for name in expected}
    for name, version in expected.items():
        if observed[name] != version:
            raise RuntimeError(f"Unexpected {name} version.")

    import torch
    import vllm
    from qwen_asr import Qwen3ASRModel

    if not torch.__version__.startswith("2.9.1"):
        raise RuntimeError("Unexpected isolated Torch version.")
    if not callable(getattr(Qwen3ASRModel, "LLM", None)):
        raise RuntimeError("Qwen vLLM model factory is unavailable.")
    if not hasattr(vllm, "LLM"):
        raise RuntimeError("vLLM LLM import is unavailable.")
    return {
        "qwen_asr": observed["qwen-asr"],
        "vllm": observed["vllm"],
        "torch": torch.__version__,
        "torch_cuda": str(torch.version.cuda),
    }


def main() -> None:
    versions = verify()
    print(
        "QWEN_ISOLATED_IMPORT_CONTRACT=PASS "
        f"qwen_asr={versions['qwen_asr']} vllm={versions['vllm']} "
        f"torch={versions['torch']} torch_cuda={versions['torch_cuda']} weights=NOT_LOADED"
    )


if __name__ == "__main__":
    main()
