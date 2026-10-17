#!/usr/bin/env bash
# Measure the false hit rate: embed each sample prompt and its negation, entity
# swap and numeric change with the real model, and report how many would be
# wrongly served at thresholds 0.75, 0.90 and 0.95.
#
# Needs no running services. Uses the cached ONNX model, downloading 79 MB the
# first time.
source "$(dirname "$0")/common.sh"
require_venv

step "adversarial probe against all-MiniLM-L6-v2"
"$PY" -m evaluation.adversarial
