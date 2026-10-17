#!/usr/bin/env bash
# Stop everything these scripts started: the gateway, Chroma, then Redis.
# Services that were already running before the scripts touched them are left
# alone. Pass --wipe to also delete Chroma's stored vectors.
source "$(dirname "$0")/common.sh"

"$ROOT/scripts/gateway-stop.sh"
"$ROOT/scripts/chroma-stop.sh" "$@"
"$ROOT/scripts/redis-stop.sh"
ok "all stopped"
