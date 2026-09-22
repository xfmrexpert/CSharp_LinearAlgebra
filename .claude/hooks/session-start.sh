#!/bin/bash
# Make the .NET SDK available to Claude Code on the web sessions.
#
# The container image has no SDK, and the usual installer is not reachable:
# dotnet-install.sh pulls from builds.dotnet.microsoft.com, which the
# environment's network policy answers with 403. Ubuntu noble-updates/universe
# carries dotnet-sdk-10.0 and archive.ubuntu.com IS reachable, so that is the
# route. apt-get update first -- the image ships a stale index whose dotnet10
# pool paths 404.
#
# Local sessions are left alone; they have their own SDK.
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

if ! command -v dotnet >/dev/null 2>&1; then
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq dotnet-sdk-10.0
fi

# Warm the NuGet cache so the first build in the session is not also the first
# restore. Locked mode matches CI, so a lockfile drift fails here rather than
# silently resolving to something else. Non-fatal: a restore failure should
# not stop the session from starting.
dotnet restore Tensile.slnx --locked-mode || true

dotnet --version
