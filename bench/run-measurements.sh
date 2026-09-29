#!/usr/bin/env bash
# Run the measurements the open items in CLAUDE.md are waiting on, each the way
# the benchmarking methodology requires, into one dated results directory.
#
#   bench/run-measurements.sh                 # everything
#   bench/run-measurements.sh --only diag     # a subset: diag, parity, complex-gemm, complex-exp
#   bench/run-measurements.sh --cpu 2         # pin to CPU 2 instead of 0
#   bench/run-measurements.sh --dry-run       # print the commands, run nothing
#
# What it encodes, so a sitting cannot get it wrong (CLAUDE.md findings 6, 7, 12):
#   - single-thread comparisons are pinned with taskset; the threaded LU phase
#     split is not, because ProcessorCount respects the affinity mask;
#   - every BenchmarkDotNet run passes --iterationCount 31, which nothing in
#     the benchmark configuration sets;
#   - every ratio worth quoting comes from rows in one benchmark class, so no
#     step here divides numbers across two runs;
#   - the environment, the commit and the BLIS dispatch are recorded beside the
#     numbers, since a figure without its conditions cannot be compared later.
#
# Run as root if you can: BenchmarkDotNet then raises process priority, which
# measured 1.4-8.4% StdDev down to ~1% on the 12700H, and turbostat can record
# frequency and throttling alongside the unpinned run.
set -euo pipefail

cpu=0
dry_run=0
only=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu) cpu="$2"; shift 2 ;;
        --only) only="$2"; shift 2 ;;
        --dry-run) dry_run=1; shift ;;
        -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1 (see --help)" >&2; exit 2 ;;
    esac
done

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

stamp="$(date +%Y-%m-%d-%H%M%S)"
out="$root/bench/results/$stamp"
diag="$root/tools/Tensile.Diagnostics/bin/Release/net10.0/tensile-diag"
bench=(dotnet run -c Release --no-build --project bench/Tensile.Benchmarks --)
pin=(taskset -c "$cpu")

wanted() { [[ -z "$only" || ",$only," == *",$1,"* ]]; }

run() {
    # run <log name> <command...>: echo the command, then run it into the log.
    local log="$1"; shift
    echo "+ $*"
    if [[ "$dry_run" == 1 ]]; then return 0; fi
    "$@" 2>&1 | tee "$out/$log"
}

if [[ "$dry_run" == 0 ]]; then
    command -v taskset >/dev/null || { echo "taskset is required (util-linux)" >&2; exit 1; }
    mkdir -p "$out"

    {
        echo "date        : $(date -Is)"
        echo "commit      : $(git rev-parse HEAD) $(git diff --quiet || echo '(dirty)')"
        echo "user        : $(id -un) (root: $([[ $EUID == 0 ]] && echo yes || echo 'no - BDN cannot raise priority'))"
        echo "pinned cpu  : $cpu"
        echo "BLIS        : ${TENSILE_BLIS_LIBRARY:-system search}"
        echo
        uname -a
        echo
        lscpu
        echo
        dotnet --info
    } > "$out/environment.txt"

    echo "building Release"
    dotnet build Tensile.slnx -c Release > "$out/build.log" 2>&1 || { tail -20 "$out/build.log"; exit 1; }
fi

# 1. tensile-diag, both ways. Pinned gives the serial LU phase split (compare
#    with the recorded 65/14/10/11); unpinned gives the threaded one, which is
#    what the Amdahl argument for LU's 45% of threaded GEMM needs. The same
#    report also carries the BLIS dispatch, the estimator and exponential
#    accuracy tables and the native-against-embedded complex report.
if wanted diag; then
    run diag-pinned.txt "${pin[@]}" "$diag"

    if [[ "$dry_run" == 0 && $EUID == 0 ]] && command -v turbostat >/dev/null; then
        turbostat --quiet --interval 1 --out "$out/turbostat-diag-unpinned.txt" &
        turbostat_pid=$!
        trap 'kill "$turbostat_pid" 2>/dev/null || true' EXIT
    fi

    run diag-unpinned.txt "$diag"

    if [[ -n "${turbostat_pid:-}" ]]; then
        kill "$turbostat_pid" 2>/dev/null || true
        trap - EXIT
    fi
fi

# 2. GEMM against BLIS, both sides in one class so the ratio is within one
#    process. Read the BLIS rows first: an unchanged binary reading 10-20% off
#    its previous figure means nothing in the run compares with another sitting.
if wanted parity; then
    run parity.txt "${pin[@]}" "${bench[@]}" --filter '*GemmVsBlis*' --iterationCount 31 \
        --artifacts "$out/parity"
    if [[ "$dry_run" == 0 ]] && grep -q "BLIS not found" "$out/parity.txt"; then
        echo "parity: SKIPPED, BLIS not found (set TENSILE_BLIS_LIBRARY)" | tee -a "$out/parity.txt"
    fi
fi

# 3. 4M against one real GEMM and the embedded route: the ratios, and whether
#    the 2^21 split-buffer cap is in the right place (n=1024 is over it).
if wanted complex-gemm; then
    run complex-gemm.txt "${pin[@]}" "${bench[@]}" --filter '*ComplexGemm*' --iterationCount 31 \
        --artifacts "$out/complex-gemm"
fi

# 4. Native complex exponentials against the embedded route.
if wanted complex-exp; then
    run complex-exp.txt "${pin[@]}" "${bench[@]}" --filter '*ComplexExponential*' --iterationCount 31 \
        --artifacts "$out/complex-exp"
fi

if [[ "$dry_run" == 0 ]]; then
    echo
    echo "results in $out"
    echo "the summary tables are the '| Method' blocks of each .txt, and the"
    echo "'=== LU phase breakdown ===' sections of diag-pinned.txt and diag-unpinned.txt"
fi
