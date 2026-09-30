"""Small timing helpers. All durations use monotonic clocks."""
from __future__ import annotations

import time
from dataclasses import dataclass


def now() -> float:
    return time.perf_counter()


def elapsed_ms(start: float, end: float | None = None) -> float:
    stop = now() if end is None else end
    return max(0.0, (stop - start) * 1000.0)


@dataclass(frozen=True)
class InferenceMetrics:
    model_inference_ms: float
    queue_wait_ms: float
    server_to_transcript_ms: float | None = None
    segment_wait_ms: float | None = None