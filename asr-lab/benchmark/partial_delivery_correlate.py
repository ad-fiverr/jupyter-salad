"""Correlate content-free partial-candidate traces from server and browser exports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


BOUNDARIES = (
    "decode_push_result",
    "runtime_candidate",
    "canonical_filter",
    "server_event",
    "ws_send",
    "browser_message_event",
    "browser_onmessage_parsed",
    "stream_events_append",
    "export_json",
)
CLASSIFICATIONS = (
    "SERVER_WS_SEND_STOP",
    "FRAME_DELIVERY_STOP",
    "BROWSER_ONMESSAGE_STOP",
    "STREAM_EVENTS_APPEND_STOP",
    "EXPORT_OMISSION",
)
FILTER_REJECTION_OUTCOMES = frozenset({
    "stale_fence",
    "invalid_scope",
    "classify_error",
    "invalid_candidate",
    "stitch_rejected",
    "unbound_fence",
})


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _last_record(events: list[dict[str, Any]], boundary: str) -> dict[str, Any] | None:
    matching = [event for event in events if event.get("boundary") == boundary]
    if boundary == "ws_send":
        completed = [event for event in matching if event.get("phase") == "post"]
        return (completed or matching)[-1] if matching else None
    return matching[-1] if matching else None


def correlate(server_path: Path, browser_path: Path) -> dict[str, Any]:
    server_payload = json.loads(server_path.read_text(encoding="utf-8"))
    browser_payload = json.loads(browser_path.read_text(encoding="utf-8"))
    if not isinstance(server_payload, dict) or not isinstance(browser_payload, dict):
        raise ValueError("both input files must contain JSON objects")
    server_events = server_payload.get("events", [])
    browser_streaming = browser_payload.get("streaming", {})
    if not isinstance(browser_streaming, dict):
        raise ValueError("browser streaming result must be a JSON object")
    browser_trace = browser_streaming.get("partial_delivery_diagnostics") or {}
    if not isinstance(browser_trace, dict):
        raise ValueError("browser partial-delivery diagnostics must be a JSON object")
    browser_partials = browser_streaming.get("partials", [])
    browser_events = browser_trace.get("events", [])
    if not isinstance(server_events, list) or not isinstance(browser_events, list) or not isinstance(browser_partials, list):
        raise ValueError("trace events and browser partials must be JSON arrays")
    if not all(isinstance(row, dict) for row in server_events + browser_events):
        raise ValueError("every trace event must be a JSON object")

    stream_id = server_payload.get("stream_id") or browser_streaming.get("public_stream_id")
    if not isinstance(stream_id, str) or not stream_id:
        raise ValueError("a public stream_id is required in the server trace or browser export")
    server_events = [row for row in server_events if row.get("stream_id") == stream_id]
    browser_events = [row for row in browser_events if row.get("stream_id") == stream_id]
    events = server_events + browser_events
    export_ids = {
        f"{partial.get('public_stream_id') or partial.get('stream_id')}/{partial.get('epoch_seq')}/{partial.get('revision')}"
        for partial in browser_partials
        if isinstance(partial, dict)
        and isinstance(partial.get("public_stream_id") or partial.get("stream_id"), str)
        and type(partial.get("epoch_seq")) is int
        and type(partial.get("revision")) is int
    }
    expected_export_ids = export_ids | {
        row["partial_id"] for row in browser_events
        if row.get("boundary") == "export_json"
        and row.get("outcome") == "retained_partial_omitted_from_export_array"
        and isinstance(row.get("partial_id"), str)
    }

    latest = {}
    for boundary in BOUNDARIES:
        row = _last_record(events, boundary)
        latest[boundary] = None if row is None else {
            key: row.get(key) for key in (
                "partial_id", "utc_timestamp", "monotonic_timestamp_ms", "text_utf8_bytes",
                "text_hash_fnv1a32", "language", "phase", "outcome", "export_id",
                "filter_outcome", "filter_code", "cursor_end_samples", "cursor_ms", "audio_cursor_ms",
                "decode_steps_delta", "revision_previous", "revision_new",
            ) if key in row
        }

    classification = "UNKNOWN"
    discontinuity = None
    decode_rows = [row for row in server_events if row.get("boundary") == "decode_push_result"]
    raw_candidate_rows = [
        row for row in server_events
        if row.get("boundary") == "runtime_candidate"
        and isinstance(row.get("partial_id"), str)
    ]
    filter_rows = [row for row in server_events if row.get("boundary") == "canonical_filter"]

    if classification == "UNKNOWN" and discontinuity is None and (decode_rows or filter_rows):
        for raw_row in raw_candidate_rows:
            partial_id = raw_row["partial_id"]
            outcomes = [row for row in filter_rows if row.get("partial_id") == partial_id]
            if not outcomes:
                discontinuity = {
                    "type": "raw_candidate_without_filter_outcome",
                    "partial_id": partial_id,
                    "scope": "filter outcome was not observed; no rejection or acceptance is inferred",
                }
                break
            filter_row = outcomes[-1]
            filter_outcome = filter_row.get("filter_outcome")
            if isinstance(filter_outcome, str) and filter_outcome in FILTER_REJECTION_OUTCOMES:
                classification = "RAW_CANDIDATE_REJECTED"
                discontinuity = {
                    "type": "canonical_filter_rejected_raw_candidate",
                    "partial_id": partial_id,
                    "filter_outcome": filter_outcome,
                    "filter_code": filter_row.get("filter_code"),
                }
                break
            if filter_outcome != "accepted":
                discontinuity = {
                    "type": "invalid_or_missing_filter_outcome",
                    "partial_id": partial_id,
                    "scope": "filter outcome is absent or unrecognized; no rejection or acceptance is inferred",
                }
                break
            if not any(
                row.get("boundary") == "server_event" and row.get("partial_id") == partial_id
                for row in server_events
            ):
                classification = "ACCEPTED_CANDIDATE_NO_B2"
                discontinuity = {
                    "type": "accepted_candidate_without_server_event",
                    "partial_id": partial_id,
                }
                break

    if classification == "UNKNOWN" and discontinuity is None and filter_rows and not decode_rows:
        discontinuity = {
            "type": "canonical_filter_without_decode_opportunity_trace",
            "scope": "new-format candidate boundaries exist but B0 was not observed",
        }

    if classification == "UNKNOWN" and discontinuity is None and decode_rows:
        current_decode = decode_rows[-1]
        previous_decode = decode_rows[-2] if len(decode_rows) > 1 else None
        cursor_advanced = (
            previous_decode is not None
            and type(current_decode.get("cursor_end_samples")) is int
            and type(previous_decode.get("cursor_end_samples")) is int
            and current_decode["cursor_end_samples"] > previous_decode["cursor_end_samples"]
        )
        if (
            cursor_advanced
            and type(current_decode.get("decode_steps_delta")) is int
            and current_decode["decode_steps_delta"] > 0
            and current_decode.get("outcome") in {"empty_text", "unchanged_text"}
        ):
            classification = "DECODE_CONTINUES_NO_TEXT_CHANGE"
            discontinuity = {
                "type": "decode_cursor_advanced_without_changed_text_candidate",
                "cursor_from_samples": previous_decode["cursor_end_samples"],
                "cursor_to_samples": current_decode["cursor_end_samples"],
                "decode_outcome": current_decode.get("outcome"),
                "last_candidate_partial_id": (latest["runtime_candidate"] or {}).get("partial_id"),
            }
        else:
            changed_decode = next(
                (row for row in reversed(decode_rows)
                 if row.get("outcome") == "changed_text_candidate"),
                None,
            )
            if changed_decode is not None:
                changed_id = changed_decode.get("partial_id")
                if not isinstance(changed_id, str):
                    discontinuity = {
                        "type": "changed_decode_missing_candidate_identity",
                        "scope": "changed B0 result could not be mapped to a public epoch/revision identity",
                    }
                elif not any(row.get("partial_id") == changed_id for row in raw_candidate_rows):
                    discontinuity = {
                        "type": "changed_decode_without_raw_candidate_observation",
                        "partial_id": changed_id,
                        "scope": "absence of B1 is insufficient to locate candidate construction; retain UNKNOWN",
                    }

    if classification == "UNKNOWN" and discontinuity is None and not decode_rows and not filter_rows:
        # In the first instrumentation iteration B1 was recorded after the
        # result gate. Preserve that legacy interpretation for saved traces.
        for raw_row in raw_candidate_rows:
            partial_id = raw_row["partial_id"]
            if not any(
                row.get("boundary") == "server_event" and row.get("partial_id") == partial_id
                for row in server_events
            ):
                classification = "SERVER_EVENT_PREPARATION_STOP"
                discontinuity = {
                    "type": "identity_missing",
                    "after_boundary": "runtime_candidate",
                    "before_boundary": "server_event",
                    "partial_id": partial_id,
                    "scope": "legacy trace predates the explicit canonical-filter boundary",
                }
                break

    if classification == "UNKNOWN" and discontinuity is None:
        delivery_boundaries = (
            "server_event", "ws_send", "browser_message_event",
            "browser_onmessage_parsed", "stream_events_append", "export_json",
        )
        for index, classification_name in enumerate(CLASSIFICATIONS):
            upstream = delivery_boundaries[index]
            downstream = delivery_boundaries[index + 1]
            upstream_rows = [
                row for row in events
                if row.get("boundary") == upstream and isinstance(row.get("partial_id"), str)
                and (upstream != "ws_send" or (row.get("phase") == "post" and row.get("outcome") == "sent"))
                and (downstream != "export_json" or row.get("partial_id") in expected_export_ids)
            ]
            downstream_rows = [row for row in events if row.get("boundary") == downstream]
            for upstream_row in upstream_rows:
                upstream_id = upstream_row["partial_id"]
                matching_rows = [row for row in downstream_rows if row.get("partial_id") == upstream_id]
                if downstream == "ws_send":
                    completed = [row for row in matching_rows if row.get("phase") == "post"]
                    downstream_row = completed[-1] if completed else (matching_rows[-1] if matching_rows else None)
                    reached = downstream_row is not None and downstream_row.get("outcome") == "sent"
                elif downstream == "export_json":
                    downstream_row = matching_rows[-1] if matching_rows else None
                    reached = downstream_row is not None and downstream_row.get("outcome") == "included"
                else:
                    downstream_row = matching_rows[-1] if matching_rows else None
                    reached = downstream_row is not None
                if not reached:
                    classification = classification_name
                    discontinuity = {
                        "type": "identity_missing",
                        "after_boundary": upstream,
                        "before_boundary": downstream,
                        "partial_id": upstream_id,
                        "downstream_last_partial_id": (latest[downstream] or {}).get("partial_id"),
                        "downstream_outcome": None if downstream_row is None else downstream_row.get("outcome"),
                        "scope": "first missing identity in the retained upstream boundary sequence",
                    }
                    break
                content_fields = ("text_utf8_bytes", "text_hash_fnv1a32", "language")
                changed_fields = [
                    key for key in content_fields
                    if upstream_row.get(key) is not None and downstream_row.get(key) is not None
                    and upstream_row[key] != downstream_row[key]
                ]
                if changed_fields:
                    classification = "UNKNOWN"
                    discontinuity = {
                        "type": "same_identity_content_mismatch",
                        "after_boundary": upstream,
                        "before_boundary": downstream,
                        "partial_id": upstream_id,
                        "changed_fields": changed_fields,
                    }
                    break
            if discontinuity is not None:
                break

    latest_ids = [
        (latest[boundary] or {}).get("partial_id") for boundary in BOUNDARIES[1:]
    ]
    if classification == "UNKNOWN" and discontinuity is None and all(isinstance(value, str) for value in latest_ids):
        if len(set(latest_ids)) == 1:
            classification = "PARTIAL_GAP_NOT_REPRODUCED"
        else:
            classification = "UNKNOWN"

    server_dropped_value = server_payload.get("dropped_event_count", 0)
    browser_dropped_value = browser_trace.get("dropped_event_count", 0)
    server_dropped = server_dropped_value if type(server_dropped_value) is int and server_dropped_value >= 0 else None
    browser_dropped = browser_dropped_value if type(browser_dropped_value) is int and browser_dropped_value >= 0 else None
    trace_truncated = server_dropped is None or browser_dropped is None or bool(server_dropped or browser_dropped)
    if trace_truncated and classification != "UNKNOWN":
        observed_discontinuity = discontinuity
        classification = "UNKNOWN"
        discontinuity = {
            "type": "incomplete_trace",
            "observed_discontinuity": observed_discontinuity,
            "scope": "bounded trace rows were dropped or completeness metadata was invalid; no causal classification is safe",
        }

    send_rows = [row for row in server_events if row.get("boundary") == "ws_send"]
    return {
        "schema_version": 1,
        "stream_id": stream_id,
        "classification": classification,
        "first_discontinuity": discontinuity,
        "last_partial_at_each_boundary": latest,
        "send_outcomes": {
            "sent": sum(row.get("outcome") == "sent" for row in send_rows),
            "errors": sum(str(row.get("outcome", "")).startswith("error:") for row in send_rows),
            "pending_pre_send": sum(row.get("phase") == "pre" for row in send_rows),
        },
        "trace_completeness": {
            "server_dropped_event_count": server_dropped,
            "browser_dropped_event_count": browser_dropped,
            "truncated": trace_truncated,
        },
        "interpretation_limit": (
            "B0 can show decode pushes that advanced the cursor without changed text. A changed B0 without matching B1/filter evidence remains UNKNOWN. "
            "PARTIAL_GAP_NOT_REPRODUCED means only that the latest observed partial identity reached every delivery boundary."
        ),
        "inputs": {
            "server_trace": {"path": str(server_path), "sha256": _sha256(server_path)},
            "browser_export": {"path": str(browser_path), "sha256": _sha256(browser_path)},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-trace", required=True, type=Path)
    parser.add_argument("--browser-export", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = correlate(args.server_trace, args.browser_export)
    serialized = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        args.output.write_text(serialized, encoding="utf-8")


if __name__ == "__main__":
    main()
