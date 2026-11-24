#!/usr/bin/env bash
# Stop the container stack, keeping the cached data unless --wipe is given.
#
#   scripts/compose-down.sh          stop and remove the containers; volumes stay
#   scripts/compose-down.sh --wipe   also delete the Redis and Chroma volumes
source "$(dirname "$0")/common.sh"

case "${1:-}" in
  "")
    step "stopping the containers (cached data is kept)"
    compose down
    ;;
  --wipe)
    step "stopping the containers and deleting the cached data"
    compose down --volumes
    ;;
  *)
    die "unknown option '$1'; the only option is --wipe"
    ;;
esac
ok "container stack stopped"
