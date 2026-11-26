"""The benchmark report: reads harness records, compares runs to the baseline, draws the chart."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks import report


def _record(label: str, mode: str, threshold: float | None, cost: float, hit_ratio: float,
            false_rate: float | None, negation: float | None = None) -> dict[str, Any]:
    by_kind = {} if negation is None else {"negation": negation, "paraphrase": 0.0}
    hit = None if mode == "direct" else 14.0
    return {
        "label": label, "mode": mode, "threshold": threshold, "commit": "abc1234",
        "simulated": True,
        "summary": {
            "requests": 200, "errors": 0, "hit_ratio": hit_ratio,
            "false_hit_rate": false_rate, "false_hit_rate_by_kind": by_kind,
            "provider_calls": 200 if mode == "direct" else 60, "cost_usd": cost,
            "latency_ms": {"hit_p50": hit, "hit_p95": hit, "miss_p50": 800.0,
                           "miss_p95": 820.0},
        },
    }


DIRECT = _record("direct", "direct", None, 0.002, 0.0, None)
STRICT = _record("t0.95", "gateway", 0.95, 0.001, 0.5, 0.05, negation=0.4)
LOOSE = _record("t0.75", "gateway", 0.75, 0.0005, 0.75, 0.25, negation=1.0)


def _files(tmp_path: Path, *records: dict[str, Any]) -> list[Path]:
    paths = []
    for record in records:
        path = tmp_path / f"{record['label']}.json"
        path.write_text(json.dumps(record))
        paths.append(path)
    return paths


def test_runs_are_ordered_baseline_first_then_by_threshold(tmp_path: Path) -> None:
    runs = report.load_runs(_files(tmp_path, STRICT, LOOSE, DIRECT))
    assert [run["label"] for run in runs] == ["direct", "t0.75", "t0.95"]


def test_a_file_that_is_not_a_harness_record_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "other.json"
    path.write_text(json.dumps({"label": "x"}))
    with pytest.raises(ValueError, match="missing"):
        report.load_runs([path])


def test_no_runs_is_an_error() -> None:
    with pytest.raises(ValueError, match="no runs"):
        report.load_runs([])


def test_cost_saved_is_measured_against_the_direct_run() -> None:
    rows = {row["run"]: row for row in report.table([DIRECT, LOOSE, STRICT])}
    assert rows["direct"]["cost saved"] == "0.0%"
    assert rows["t0.75"]["cost saved"] == "75.0%"
    assert rows["t0.95"]["cost saved"] == "50.0%"


def test_without_a_baseline_cost_saved_is_not_invented() -> None:
    (row,) = report.table([STRICT])
    assert row["cost saved"] == "n/a"


def test_every_hit_ratio_sits_beside_its_false_hit_rate() -> None:
    row = {row["run"]: row for row in report.table([DIRECT, LOOSE])}["t0.75"]
    assert (row["hit ratio"], row["false hit rate"], row["negation false hits"]) == (
        "75.0%", "25.0%", "100.0%",
    )
    assert row["hit p50 / p95 ms"] == "14 / 14"
    assert row["miss p50 / p95 ms"] == "800 / 820"


def test_the_markdown_is_a_table_with_provenance_and_the_simulation_note() -> None:
    text = report.markdown([DIRECT, LOOSE, STRICT], chart="report.svg")
    lines = text.splitlines()
    assert "Commit: abc1234" in text
    assert any(set(line.replace(" ", "")) == {"|", "-"} for line in lines)
    assert sum(line.startswith("| t0.") for line in lines) == 2
    assert "simulated" in text
    assert "](report.svg)" in text


def test_a_run_against_a_real_provider_is_not_labelled_simulated() -> None:
    real = dict(DIRECT, simulated=False)
    assert "simulated" not in report.markdown([real])


def test_the_chart_draws_two_bars_for_each_gateway_run_only() -> None:
    chart = report.svg([DIRECT, LOOSE, STRICT])
    assert chart.startswith("<svg")
    lines = chart.splitlines()
    bars = [line for line in lines if line.startswith("<rect x=") and "<text" not in line]
    assert len(bars) == 4  # hit ratio and false hit rate for each of the two gateway runs
    assert "75%" in chart and "25%" in chart and "50%" in chart and "5%" in chart
    assert ">direct<" not in chart


def test_main_writes_the_report_and_its_chart(tmp_path: Path) -> None:
    out = tmp_path / "report.md"
    report.main([*map(str, _files(tmp_path, DIRECT, LOOSE)), "--out", str(out)])
    assert out.read_text().startswith("# Benchmark report")
    assert (tmp_path / "report.svg").read_text().startswith("<svg")


def test_the_summary_is_plain_numbers_with_savings_against_the_baseline() -> None:
    summary = report.summary([DIRECT, LOOSE, STRICT])
    assert summary["simulated"] is True and summary["commits"] == ["abc1234"]
    rows = {row["label"]: row for row in summary["runs"]}
    assert rows["t0.75"]["hit_ratio"] == 0.75
    assert rows["t0.75"]["cost_saved"] == pytest.approx(0.75)
    assert rows["t0.75"]["negation_false_hit_rate"] == 1.0
    assert rows["direct"]["hit_p50_ms"] is None and rows["direct"]["miss_p50_ms"] == 800.0
    assert report.summary([STRICT])["runs"][0]["cost_saved"] is None


def test_main_can_also_write_the_summary_as_json(tmp_path: Path) -> None:
    out, numbers = tmp_path / "report.md", tmp_path / "numbers.json"
    report.main([*map(str, _files(tmp_path, DIRECT, LOOSE)), "--out", str(out),
                 "--json", str(numbers)])
    assert json.loads(numbers.read_text())["runs"][1]["label"] == "t0.75"
