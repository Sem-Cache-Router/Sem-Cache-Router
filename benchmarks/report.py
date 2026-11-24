# Benchmark report generation.
#
# Holds:
#   load_runs(), which reads the JSON records the harness writes
#   table(), the comparison of every run against the no cache baseline
#   markdown(), the report itself, stamped with the commits the runs came from
#   svg(), a chart of hit ratio beside false hit rate for each threshold
#
# Every headline figure is printed next to the accuracy it came at, because a
# cost number without an accuracy number beside it is not a result. The chart is
# written as plain SVG rather than through a plotting library, so producing a
# report adds no dependency to the project.

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REQUIRED = ("label", "mode", "summary", "commit", "simulated")
SVG_WIDTH, SVG_HEIGHT, BAR_AREA = 560, 300, 220


def load_runs(paths: list[Path]) -> list[dict[str, Any]]:
    """Read harness records, the direct baseline first, then gateway runs by threshold."""
    runs = []
    for path in paths:
        record = json.loads(path.read_text())
        missing = [key for key in REQUIRED if key not in record]
        if missing:
            raise ValueError(f"{path} is not a harness record: missing {missing}")
        runs.append(record)
    if not runs:
        raise ValueError("no runs to report")
    return sorted(runs, key=lambda r: (r["mode"] != "direct", r.get("threshold") or 0.0))


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f}"


def table(runs: list[dict[str, Any]]) -> list[dict[str, str]]:
    """One row per run. Savings are measured against the direct run, if there is one."""
    baseline = next((r["summary"]["cost_usd"] for r in runs if r["mode"] == "direct"), None)
    rows = []
    for run in runs:
        s, latency = run["summary"], run["summary"]["latency_ms"]
        saved = None
        if baseline:
            saved = 1 - s["cost_usd"] / baseline
        rows.append({
            "run": run["label"],
            "threshold": "-" if run.get("threshold") is None else f"{run['threshold']:.2f}",
            "hit ratio": _pct(s["hit_ratio"]),
            "false hit rate": _pct(s["false_hit_rate"]),
            "negation false hits": _pct(s["false_hit_rate_by_kind"].get("negation")),
            "provider calls": str(s["provider_calls"]),
            "cost saved": _pct(saved),
            "hit p50 / p95 ms": f"{_ms(latency['hit_p50'])} / {_ms(latency['hit_p95'])}",
            "miss p50 / p95 ms": f"{_ms(latency['miss_p50'])} / {_ms(latency['miss_p95'])}",
            "errors": str(s["errors"]),
        })
    return rows


def markdown(runs: list[dict[str, Any]], chart: str | None = None) -> str:
    """The report as Markdown: provenance, the table, and an honest note on what it is."""
    rows = table(runs)
    headers = list(rows[0])
    commits = sorted({run["commit"] for run in runs})
    lines = [
        "# Benchmark report",
        "",
        f"Runs: {len(runs)}. Commit: {', '.join(commits)}. "
        f"Requests per run: {runs[0]['summary']['requests']}.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("-" for _ in headers) + " |",
    ]
    lines += ["| " + " | ".join(row[h] for h in headers) + " |" for row in rows]
    lines += ["", "A false hit is a cache hit that served an answer produced for a prompt"
              " of a different group. Cost saved is against the direct run, which sends"
              " every prompt to the vendor with no gateway."]
    if any(run["simulated"] for run in runs):
        lines += ["", "These runs used the mock vendor API: latency and cost are simulated"
                  " and show how the gateway behaves, not what a real provider would charge."]
    if chart:
        lines += ["", f"![Hit ratio and false hit rate by threshold]({chart})"]
    return "\n".join(lines) + "\n"


def svg(runs: list[dict[str, Any]]) -> str:
    """Hit ratio and false hit rate side by side for every gateway run."""
    gateway = [run for run in runs if run["mode"] == "gateway"]
    slot = (SVG_WIDTH - 60) / max(1, len(gateway))
    bar = min(40.0, slot / 3)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" '
        'font-family="sans-serif" font-size="12">',
        f'<rect width="{SVG_WIDTH}" height="{SVG_HEIGHT}" fill="white"/>',
        f'<line x1="40" y1="{BAR_AREA + 20}" x2="{SVG_WIDTH - 10}" y2="{BAR_AREA + 20}" '
        'stroke="#444"/>',
        '<rect x="50" y="6" width="10" height="10" fill="#2a7ab0"/>'
        '<text x="64" y="15">hit ratio</text>',
        '<rect x="140" y="6" width="10" height="10" fill="#c0392b"/>'
        '<text x="154" y="15">false hit rate</text>',
    ]
    for index, run in enumerate(gateway):
        summary = run["summary"]
        x = 50 + index * slot
        for offset, value, colour in (
            (0.0, summary["hit_ratio"] or 0.0, "#2a7ab0"),
            (bar, summary["false_hit_rate"] or 0.0, "#c0392b"),
        ):
            height = value * BAR_AREA
            top = BAR_AREA + 20 - height
            parts.append(f'<rect x="{x + offset:.1f}" y="{top:.1f}" width="{bar:.1f}" '
                         f'height="{height:.1f}" fill="{colour}"/>')
            parts.append(f'<text x="{x + offset:.1f}" y="{top - 3:.1f}">{value * 100:.0f}%</text>')
        parts.append(f'<text x="{x:.1f}" y="{BAR_AREA + 36}">{run["label"]}</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the benchmark report from harness runs.")
    parser.add_argument("runs", nargs="+", type=Path, help="JSON records written by the harness")
    parser.add_argument("--out", type=Path, required=True, help="Markdown file to write")
    args = parser.parse_args(argv)

    runs = load_runs(args.runs)
    chart = args.out.with_suffix(".svg")
    chart.write_text(svg(runs))
    args.out.write_text(markdown(runs, chart=chart.name))
    print(f"wrote {args.out} and {chart}")


if __name__ == "__main__":
    main()
