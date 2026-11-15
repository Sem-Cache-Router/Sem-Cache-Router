#!/usr/bin/env bash
# Stop the mock vendor API that mock-upstream-start.sh started.
source "$(dirname "$0")/common.sh"
stop_pid mock
