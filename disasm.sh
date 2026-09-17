#!/usr/bin/env bash
# Dump JIT disassembly for the micro-kernels and fail on accumulator spills.
#
# This exists because of a failure mode that is silent: RyuJIT's register
# allocator is shape-sensitive, and a kernel that looks identical in source can
# have every accumulator spilled to the stack, costing roughly 5x, with no
# warning of any kind. The only reliable detector is to read the generated code.
#
# Two details matter and are easy to get wrong:
#
#   * Run the built binary directly, NOT via `dotnet run`, and not via the
#     benchmark project. The SDK driver, and BenchmarkDotNet's generated child
#     processes, are separate processes that also honour DOTNET_JitStdOutFile,
#     so they clobber each other's output and kernels go missing from the dump.
#     tools/Tensile.Diagnostics exists to be a single process that calls every
#     supported kernel and then exits.
#
#   * Scope the spill search to the kernels. The dump is filtered by method
#     name, and plenty of unrelated framework methods are also called Execute.
#
# Spill slots are rbp-relative here, not rsp-relative; both are matched.

set -euo pipefail

CONFIGURATION="${CONFIGURATION:-Release}"
OUT="${1:-kernel.asm}"
PROJECT="tools/Tensile.Diagnostics/Tensile.Diagnostics.csproj"
BINARY="${BINARY:-./tools/Tensile.Diagnostics/bin/${CONFIGURATION}/net10.0/tensile-diag}"

if [[ ! -x "$BINARY" ]]; then
    echo "building $CONFIGURATION first ($BINARY not found)"
    dotnet build -c "$CONFIGURATION" "$PROJECT" >/dev/null
fi

# The JIT appends to an existing dump file rather than replacing it, so a
# second run against the same path would report every kernel twice and double
# every count below. Start from an empty file.
: > "$OUT"

DOTNET_JitDisasm="Execute" \
DOTNET_JitStdOutFile="$OUT" \
DOTNET_TieredCompilation=0 \
DOTNET_ReadyToRun=0 \
"$BINARY" --quiet >/dev/null

echo "disassembly written to $OUT"
echo

# Split the dump into one file per method and inspect only the kernels.
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

awk -v dir="$WORK" '
    /^; Assembly listing for method/ {
        name = $0
        sub(/^; Assembly listing for method /, "", name)
        sub(/\(.*$/, "", name)
        gsub(/[^A-Za-z0-9_.:]/, "", name)
        file = dir "/" name ".txt"
        next
    }
    file { print > file }
' "$OUT"

SPILL_PATTERN='vmov(ups|apd|upd) +(zmm|ymm)word ptr \[(rbp|rsp)'
FOUND=0
FAILED=0

printf '%-30s %8s %8s\n' "kernel" "FMAs" "spills"

for listing in "$WORK"/Tensile.Kernels.*Kernel*Execute.txt; do
    [[ -e "$listing" ]] || continue

    FOUND=$((FOUND + 1))
    name="$(basename "$listing" .txt)"
    name="${name#Tensile.Kernels.}"

    fmas=$(grep -c "vfmadd" "$listing" || true)
    spills=$(grep -cE "$SPILL_PATTERN" "$listing" || true)

    printf '%-30s %8s %8s\n' "$name" "$fmas" "$spills"

    if [[ "$spills" -ne 0 ]]; then
        echo
        echo "FAIL: $name spills accumulators to the stack. Offending instructions:"
        grep -nE "$SPILL_PATTERN" "$listing" | head -20
        FAILED=1
    fi
done

echo

if [[ "$FOUND" -eq 0 ]]; then
    echo "FAIL: no micro-kernel disassembly was captured."
    echo "Nothing was verified. Check that $BINARY runs and that the JIT dump is reaching $OUT."
    exit 1
fi

if [[ "$FAILED" -ne 0 ]]; then
    exit 1
fi

echo "OK: $FOUND micro-kernel(s) inspected, no accumulator spills."
