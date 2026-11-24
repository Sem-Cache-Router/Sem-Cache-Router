#!/usr/bin/env bash
# Build and start the whole stack in containers: gateway, Redis, Chroma and Jaeger.
#
# This is the deployment path. run-tier1.sh and run-tier2.sh run the gateway
# from the virtualenv instead. Both publish the gateway on port 8000, so stop
# one before starting the other. Cache state lives in named volumes and
# survives scripts/compose-down.sh unless it is given --wipe.
source "$(dirname "$0")/common.sh"

[ "$GATEWAY_PORT" = 8000 ] || die "the Compose stack publishes the gateway on 8000; unset GATEWAY_PORT"

if gateway_up && ! compose ps --status running --services 2>/dev/null | grep -qx gateway; then
  die "a gateway outside Compose already holds port 8000; stop it with scripts/stop-all.sh"
fi

step "building the image and starting the containers"
compose up -d --build

step "waiting for the gateway"
wait_for 180 "the gateway" gateway_up
ok "health: $(curl -s "$GATEWAY_URL/health")"
echo
echo "Gateway:   $GATEWAY_URL"
echo "Jaeger UI: http://localhost:16686"
echo "Stop with scripts/compose-down.sh (add --wipe to delete the cached data too)."
