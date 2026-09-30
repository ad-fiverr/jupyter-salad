"""Normalize authored SAPI fixtures and add deterministic, seeded background noise."""
from __future__ import annotations

import math
import random
import struct
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "fixtures"
SAMPLE_RATE = 16_000


def load_pcm(path: Path) -> list[int]:
    with wave.open(str(path), "rb") as handle:
        if (handle.getnchannels(), handle.getsampwidth(), handle.getframerate(), handle.getcomptype()) != (1, 2, SAMPLE_RATE, "NONE"):
            raise ValueError(f"unexpected fixture format: {path.name}")
        raw = handle.readframes(handle.getnframes())
    return [item[0] for item in struct.iter_unpack("<h", raw)]


def write_pcm(path: Path, values: list[int]) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(b"".join(struct.pack("<h", max(-32768, min(32767, value))) for value in values))


def main() -> None:
    for path in sorted(ROOT.glob("*.wav")):
        if path.stem.endswith("_ruido_moderado"):
            continue
        values = load_pcm(path)
        values.extend([0] * SAMPLE_RATE)  # one second of end silence for endpointing.
        write_pcm(path, values)
    source = ROOT / "frase_larga.wav"
    speech = load_pcm(source)
    energy = sum(value * value for value in speech) / max(1, len(speech))
    speech_rms = math.sqrt(energy)
    noise_rms = speech_rms / (10 ** (18 / 20))
    rng = random.Random(29092026)
    mixed = [round(sample + rng.gauss(0, noise_rms)) for sample in speech]
    mixed.extend([0] * SAMPLE_RATE)
    output = ROOT / "frase_larga_ruido_moderado.wav"
    write_pcm(output, mixed)
    (ROOT / "frase_larga_ruido_moderado.txt").write_text(
        (ROOT / "frase_larga.txt").read_text(encoding="utf-8"), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
