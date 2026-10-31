from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "benchmark" / "partial_delivery_correlate.py"
SPEC = importlib.util.spec_from_file_location("partial_delivery_correlate", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
CORRELATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CORRELATOR)

BOUNDARIES = [
    "decode_push_result",
    "runtime_candidate",
    "canonical_filter",
    "server_event",
    "ws_send",
    "browser_message_event",
    "browser_onmessage_parsed",
    "stream_events_append",
    "export_json",
]


class PartialDeliveryCorrelatorTests(unittest.TestCase):
    def run_case(
        self,
        missing_boundary: str | None = None,
        *,
        send_error: bool = False,
        mismatch_boundary: str | None = None,
        filter_outcome: str | None = "accepted",
        filter_outcome_present: bool = True,
        server_dropped: int = 0,
        browser_dropped: int = 0,
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            server_events = []
            browser_events = []
            for boundary in BOUNDARIES:
                if boundary == missing_boundary:
                    continue
                row = {
                    "boundary": boundary,
                    "partial_id": "public/3/9",
                    "stream_id": "public",
                    "utc_timestamp": "2026-10-30T05:00:00Z",
                    "monotonic_timestamp_ms": 1234.5,
                    "text_utf8_bytes": 14,
                    "text_hash_fnv1a32": "1234abcd",
                    "language": "Spanish",
                }
                if boundary == "decode_push_result":
                    row.update({
                        "outcome": "changed_text_candidate",
                        "cursor_end_samples": 16_000,
                        "cursor_ms": 1000,
                        "decode_steps_delta": 1,
                        "revision_previous": 8,
                        "revision_new": 9,
                    })
                if boundary == "canonical_filter":
                    if filter_outcome_present:
                        row["filter_outcome"] = filter_outcome
                    if isinstance(filter_outcome, str) and filter_outcome in CORRELATOR.FILTER_REJECTION_OUTCOMES:
                        row["filter_code"] = filter_outcome
                if boundary == "export_json":
                    row["outcome"] = "included"
                if boundary == mismatch_boundary:
                    row["text_hash_fnv1a32"] = "deadbeef"
                if boundary == "ws_send":
                    server_events.append({**row, "phase": "pre", "outcome": "pending"})
                    server_events.append({
                        **row, "phase": "post",
                        "outcome": "error:RuntimeError" if send_error else "sent",
                    })
                elif boundary in BOUNDARIES[:4]:
                    server_events.append(row)
                else:
                    browser_events.append(row)

            server_path = root / "server.json"
            browser_path = root / "browser.json"
            server_path.write_text(json.dumps({
                "stream_id": "public", "dropped_event_count": server_dropped, "events": server_events,
            }), encoding="utf-8")
            browser_path.write_text(json.dumps({
                "streaming": {
                    "public_stream_id": "public",
                    "partials": [{"stream_id": "public", "epoch_seq": 3, "revision": 9, "text": "not copied into trace"}],
                    "partial_delivery_diagnostics": {
                        "dropped_event_count": browser_dropped, "events": browser_events,
                    },
                },
            }), encoding="utf-8")
            return CORRELATOR.correlate(server_path, browser_path)

    def test_all_observed_boundaries_keep_the_same_identity(self):
        result = self.run_case()
        self.assertEqual(result["classification"], "PARTIAL_GAP_NOT_REPRODUCED")
        self.assertIsNone(result["first_discontinuity"])
        self.assertEqual(
            {row["partial_id"] for row in result["last_partial_at_each_boundary"].values()},
            {"public/3/9"},
        )
        self.assertEqual(len(result["inputs"]["server_trace"]["sha256"]), 64)
        self.assertNotIn("not copied into trace", json.dumps(result))

    def test_each_first_missing_boundary_maps_to_its_own_classification(self):
        expected = {
            "server_event": "ACCEPTED_CANDIDATE_NO_B2",
            "ws_send": "SERVER_WS_SEND_STOP",
            "browser_message_event": "FRAME_DELIVERY_STOP",
            "browser_onmessage_parsed": "BROWSER_ONMESSAGE_STOP",
            "stream_events_append": "STREAM_EVENTS_APPEND_STOP",
            "export_json": "EXPORT_OMISSION",
        }
        for boundary, classification in expected.items():
            with self.subTest(boundary=boundary):
                result = self.run_case(missing_boundary=boundary)
                self.assertEqual(result["classification"], classification)
                self.assertIsNotNone(result["first_discontinuity"])

    def test_send_error_is_not_counted_as_successful_send(self):
        result = self.run_case(send_error=True)
        self.assertEqual(result["classification"], "SERVER_WS_SEND_STOP")
        self.assertEqual(result["send_outcomes"]["errors"], 1)
        self.assertEqual(result["last_partial_at_each_boundary"]["ws_send"]["outcome"], "error:RuntimeError")

    def test_missing_runtime_observation_stays_unknown(self):
        result = self.run_case(missing_boundary="runtime_candidate")
        self.assertEqual(result["classification"], "UNKNOWN")
        self.assertEqual(
            result["first_discontinuity"]["type"],
            "changed_decode_without_raw_candidate_observation",
        )

    def test_advancing_decode_cursor_without_text_change_is_distinguished(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            server_events = [
                {
                    "boundary": "decode_push_result", "stream_id": "public",
                    "cursor_end_samples": 16000, "outcome": "changed_text_candidate",
                    "decode_steps_delta": 1,
                    "partial_id": "public/3/9", "revision_new": 9,
                },
                {
                    "boundary": "runtime_candidate", "stream_id": "public",
                    "partial_id": "public/3/9", "text_utf8_bytes": 5,
                    "text_hash_fnv1a32": "1a2b3c4d", "language": "Spanish",
                },
                {
                    "boundary": "canonical_filter", "stream_id": "public",
                    "partial_id": "public/3/9", "filter_outcome": "accepted",
                },
                {
                    "boundary": "server_event", "stream_id": "public",
                    "partial_id": "public/3/9", "text_utf8_bytes": 5,
                    "text_hash_fnv1a32": "1a2b3c4d", "language": "Spanish",
                },
                {
                    "boundary": "ws_send", "stream_id": "public",
                    "partial_id": "public/3/9", "phase": "pre", "outcome": "pending",
                    "text_utf8_bytes": 5, "text_hash_fnv1a32": "1a2b3c4d", "language": "Spanish",
                },
                {
                    "boundary": "ws_send", "stream_id": "public",
                    "partial_id": "public/3/9", "phase": "post", "outcome": "sent",
                    "text_utf8_bytes": 5, "text_hash_fnv1a32": "1a2b3c4d", "language": "Spanish",
                },
                {
                    "boundary": "decode_push_result", "stream_id": "public",
                    "cursor_end_samples": 32000, "outcome": "unchanged_text",
                    "decode_steps_delta": 1,
                    "partial_id": None, "revision_previous": 9, "revision_new": 9,
                },
            ]
            server_path = root / "server.json"
            browser_path = root / "browser.json"
            browser_events = [
                {
                    "boundary": boundary, "stream_id": "public", "partial_id": "public/3/9",
                    "text_utf8_bytes": 5, "text_hash_fnv1a32": "1a2b3c4d", "language": "Spanish",
                    **({"outcome": "included"} if boundary == "export_json" else {}),
                }
                for boundary in (
                    "browser_message_event", "browser_onmessage_parsed",
                    "stream_events_append", "export_json",
                )
            ]
            server_path.write_text(json.dumps({
                "stream_id": "public", "dropped_event_count": 0, "events": server_events,
            }), encoding="utf-8")
            browser_path.write_text(json.dumps({
                "streaming": {
                    "public_stream_id": "public",
                    "partials": [{"stream_id": "public", "epoch_seq": 3, "revision": 9, "text": "words"}],
                    "partial_delivery_diagnostics": {"dropped_event_count": 0, "events": browser_events},
                },
            }), encoding="utf-8")
            result = CORRELATOR.correlate(server_path, browser_path)
            self.assertEqual(result["classification"], "DECODE_CONTINUES_NO_TEXT_CHANGE")
            self.assertEqual(result["first_discontinuity"]["cursor_from_samples"], 16000)
            self.assertEqual(result["first_discontinuity"]["cursor_to_samples"], 32000)

    def test_raw_candidate_rejection_and_missing_filter_outcome_are_separate(self):
        rejected = self.run_case(filter_outcome="stale_fence")
        self.assertEqual(rejected["classification"], "RAW_CANDIDATE_REJECTED")
        self.assertEqual(rejected["first_discontinuity"]["filter_outcome"], "stale_fence")
        self.assertEqual(rejected["first_discontinuity"]["filter_code"], "stale_fence")

        unknown = self.run_case(missing_boundary="canonical_filter")
        self.assertEqual(unknown["classification"], "UNKNOWN")
        self.assertEqual(unknown["first_discontinuity"]["type"], "raw_candidate_without_filter_outcome")

        for outcome, present in (
            (None, True),
            ("future_filter_result", True),
            ([], True),
            ({"future": "filter"}, True),
            (None, False),
        ):
            with self.subTest(outcome=outcome, present=present):
                unknown = self.run_case(
                    filter_outcome=outcome,
                    filter_outcome_present=present,
                )
                self.assertEqual(unknown["classification"], "UNKNOWN")
                self.assertEqual(
                    unknown["first_discontinuity"]["type"],
                    "invalid_or_missing_filter_outcome",
                )

    def test_each_explicit_filter_rejection_is_classified_with_its_exact_code(self):
        for outcome in sorted(CORRELATOR.FILTER_REJECTION_OUTCOMES):
            with self.subTest(outcome=outcome):
                result = self.run_case(filter_outcome=outcome)
                self.assertEqual(result["classification"], "RAW_CANDIDATE_REJECTED")
                self.assertEqual(result["first_discontinuity"]["filter_outcome"], outcome)
                self.assertEqual(result["first_discontinuity"]["filter_code"], outcome)

    def test_explicit_raw_candidate_rejection_precedes_later_no_text_decode(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            server_events = [
                {
                    "boundary": "decode_push_result", "stream_id": "public",
                    "outcome": "changed_text_candidate", "cursor_end_samples": 16000,
                    "decode_steps_delta": 1, "partial_id": "public/3/9", "revision_new": 9,
                },
                {
                    "boundary": "runtime_candidate", "stream_id": "public",
                    "partial_id": "public/3/9", "text_utf8_bytes": 5,
                    "text_hash_fnv1a32": "1a2b3c4d", "language": "Spanish",
                },
                {
                    "boundary": "canonical_filter", "stream_id": "public",
                    "partial_id": "public/3/9", "filter_outcome": "stale_fence",
                    "filter_code": "stale_fence",
                },
                {
                    "boundary": "decode_push_result", "stream_id": "public",
                    "outcome": "unchanged_text", "cursor_end_samples": 32000,
                    "decode_steps_delta": 1, "partial_id": None,
                    "revision_previous": 9, "revision_new": 9,
                },
            ]
            server_path = root / "server.json"
            browser_path = root / "browser.json"
            server_path.write_text(json.dumps({
                "stream_id": "public", "dropped_event_count": 0, "events": server_events,
            }), encoding="utf-8")
            browser_path.write_text(json.dumps({
                "streaming": {
                    "public_stream_id": "public", "partials": [],
                    "partial_delivery_diagnostics": {"dropped_event_count": 0, "events": []},
                },
            }), encoding="utf-8")

            result = CORRELATOR.correlate(server_path, browser_path)

            self.assertEqual(result["classification"], "RAW_CANDIDATE_REJECTED")
            self.assertEqual(result["first_discontinuity"]["partial_id"], "public/3/9")
            self.assertEqual(result["first_discontinuity"]["filter_outcome"], "stale_fence")

    def test_same_partial_identity_with_changed_text_fingerprint_is_unknown(self):
        result = self.run_case(mismatch_boundary="server_event")
        self.assertEqual(result["classification"], "UNKNOWN")
        self.assertEqual(result["first_discontinuity"]["type"], "same_identity_content_mismatch")
        self.assertEqual(result["first_discontinuity"]["changed_fields"], ["text_hash_fnv1a32"])

    def test_trace_drops_force_unknown_instead_of_assigning_a_boundary_cause(self):
        result = self.run_case(missing_boundary="browser_message_event", server_dropped=1)
        self.assertEqual(result["classification"], "UNKNOWN")
        self.assertEqual(result["first_discontinuity"]["type"], "incomplete_trace")
        self.assertEqual(
            result["first_discontinuity"]["observed_discontinuity"]["before_boundary"],
            "browser_message_event",
        )

        fully_observed = self.run_case(server_dropped=1)
        self.assertEqual(fully_observed["classification"], "UNKNOWN")
        self.assertEqual(fully_observed["trace_completeness"]["truncated"], True)

    def test_retained_partial_omission_from_export_array_is_classified(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            server_events = []
            browser_events = []
            for boundary in BOUNDARIES[:-1]:
                row = {
                    "boundary": boundary,
                    "partial_id": "public/3/9",
                    "stream_id": "public",
                    "text_utf8_bytes": 5,
                    "text_hash_fnv1a32": "1a2b3c4d",
                    "language": "Spanish",
                }
                if boundary == "decode_push_result":
                    revision = 9
                    row.update({
                        "outcome": "changed_text_candidate",
                        "cursor_end_samples": revision * 16000,
                        "decode_steps_delta": 1,
                        "revision_new": revision,
                    })
                if boundary == "canonical_filter":
                    row["filter_outcome"] = "accepted"
                if boundary == "ws_send":
                    server_events.extend([
                        {**row, "phase": "pre", "outcome": "pending"},
                        {**row, "phase": "post", "outcome": "sent"},
                    ])
                elif boundary in BOUNDARIES[:4]:
                    server_events.append(row)
                else:
                    browser_events.append(row)
            browser_events.append({
                "boundary": "export_json",
                "partial_id": "public/3/9",
                "stream_id": "public",
                "text_utf8_bytes": 5,
                "text_hash_fnv1a32": "1a2b3c4d",
                "language": "Spanish",
                "outcome": "retained_partial_omitted_from_export_array",
            })
            server_path = root / "server.json"
            browser_path = root / "browser.json"
            server_path.write_text(json.dumps({"stream_id": "public", "events": server_events}), encoding="utf-8")
            browser_path.write_text(json.dumps({
                "streaming": {
                    "public_stream_id": "public",
                    "partials": [],
                    "partial_delivery_diagnostics": {"events": browser_events},
                },
            }), encoding="utf-8")
            result = CORRELATOR.correlate(server_path, browser_path)
            self.assertEqual(result["classification"], "EXPORT_OMISSION")
            self.assertEqual(result["first_discontinuity"]["partial_id"], "public/3/9")

    def test_first_intermediate_gap_is_found_even_when_boundary_tails_later_match(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            server_events = []
            browser_events = []
            partial_ids = ("public/3/8", "public/3/9")
            for partial_id in partial_ids:
                for boundary in BOUNDARIES:
                    if partial_id.endswith("/8") and boundary == "server_event":
                        continue
                    row = {
                        "boundary": boundary,
                        "partial_id": partial_id,
                        "stream_id": "public",
                        "text_utf8_bytes": 5,
                        "text_hash_fnv1a32": "1a2b3c4d",
                        "language": "Spanish",
                    }
                    if boundary == "decode_push_result":
                        revision = int(partial_id.rsplit("/", 1)[1])
                        row.update({
                            "outcome": "changed_text_candidate",
                            "cursor_end_samples": revision * 16_000,
                            "decode_steps_delta": 1,
                            "revision_new": revision,
                        })
                    if boundary == "canonical_filter":
                        row["filter_outcome"] = "accepted"
                    if boundary == "ws_send":
                        server_events.extend([
                            {**row, "phase": "pre", "outcome": "pending"},
                            {**row, "phase": "post", "outcome": "sent"},
                        ])
                    elif boundary in BOUNDARIES[:4]:
                        server_events.append(row)
                    else:
                        browser_events.append(row)
            server_path = root / "server.json"
            browser_path = root / "browser.json"
            server_path.write_text(json.dumps({"stream_id": "public", "events": server_events}), encoding="utf-8")
            browser_path.write_text(json.dumps({
                "streaming": {
                    "public_stream_id": "public",
                    "partials": [
                        {"stream_id": "public", "epoch_seq": 3, "revision": revision, "text": "words"}
                        for revision in (8, 9)
                    ],
                    "partial_delivery_diagnostics": {"events": browser_events},
                },
            }), encoding="utf-8")
            result = CORRELATOR.correlate(server_path, browser_path)
            self.assertEqual(result["classification"], "ACCEPTED_CANDIDATE_NO_B2")
            self.assertEqual(result["first_discontinuity"]["partial_id"], "public/3/8")


if __name__ == "__main__":
    unittest.main()
