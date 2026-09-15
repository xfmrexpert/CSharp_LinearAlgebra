#!/usr/bin/env bash
# Dump JIT disassembly for the micro-kernels and check for accumulator spills.
set -euo pipefail

METHOD="${1:-Execute}"
OUT="${2:-kernel.asm}"

DOTNET_JitDisasm="$METHOD" \
DOTNET_JitStdOutFile="$OUT" \
DOTNET_TieredCompilation=0 \
DOTNET_ReadyToRun=0 \
dotnet run -c Release > /dev/null

echo "disassembly written to $OUT"
echo
echo "accumulator spills (want zero):"
grep -cE "vmov(ups|apd|upd) +(zmm|ymm)word ptr \[(rbp|rsp)" "$OUT" || true
echo
echo "FMA count per listing:"
grep -c "vfmadd" "$OUT" || true
