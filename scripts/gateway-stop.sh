#!/usr/bin/env bash
# Stop the gateway started by gateway-start.sh.
source "$(dirname "$0")/common.sh"
stop_pid gateway
