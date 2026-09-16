# Project context

A modern dense linear algebra library for C#, written because Math.NET Numerics
is effectively dead (last commit March 2025, last stable release 5.0.0 in April
2022) and because of a genuine open question: **can someone build a better
linear algebra library for C# using current .NET?**

The eventual target application is transient response of power transformer
windings (FEM / MTL models), which means dense complex-valued frequency-domain
matrices and matrix exponentials of matrices up to ~1000x1000. Curiosity about
the managed-vs-native question is a first-class motivation, not just a means.

## Hard constraints

- **No closed-source or proprietary-licensed dependencies.** Intel MKL is
  excluded on principle, not on performance. The test is forkability: if the
  dependency is abandoned, it must be possible to fork and carry on. OpenBLAS
  (BSD-3), reference LAPACK (BSD-3) and BLIS (BSD-3) pass; MKL does not.
- **Prefer porting over binding** where a native dependency would otherwise be
  required. Reference LAPACK is open and readable; transliterating `dgeev` etc.
  to C# is legitimate and avoids per-RID native packaging, keeps Native AOT
  single-file publish clean, and keeps the whole stack debuggable.
- Math.NET itself is MIT, so its managed provider is a legitimate correctness
  oracle and starting point even though its architecture is being discarded.

## Style

- Complete, clean, working code with solid architecture. Clear code over
  gimmicks. Explicitly rejected: expression templates via `ref struct` lazy
  nodes — they fuse temporaries but wreck debuggability for little gain on
  level-3 operations that are memory-bound anyway.
- State uncertainty explicitly rather than glossing over it. Several
  conclusions in this file are inferences from indirect evidence and are
  labelled as such.

---

# Current state

Library plus a test project, `net10.0`, column-major throughout, unit row
stride, double precision only, no transposes in GEMM, no complex yet. The
library itself still takes no package references; the test project takes
xunit.v3.

| File | Purpose |
| --- | --- |
| `MicroKernels.cs` | `IMicroKernel` + AVX-512 16x8, AVX2/FMA 8x6, scalar 4x4 |
| `Packing.cs` | A/B panel packing, zero-padded edges, panel-range variants for parallel packing |
| `Gemm.cs` | Five-loop blocked GEMM, generic over kernel, single-threaded |
| `ParallelGemm.cs` | Multi-threaded GEMM, loop-2 (jr) parallelism |
| `GemmDispatch.cs` | Owns both scratches, picks serial vs threaded by work size |
| `KernelProbe.cs` | Micro-kernel ceiling on L1-resident panels |
| `Blis.cs` | Optional native `bli_dgemm` binding + dispatch/ABI queries |
| `Reference.cs` | Naive GEMM oracle + relative Frobenius residual |
| `TimingStatistics.cs` | Median and interpolated quartiles |
| `Program.cs` | Correctness checks and benchmarks; exits non-zero on failure |
| `Blas1.cs` | Vectorised scal/axpy/iamax/dot for the LU panel |
| `Blas2.cs` | A*X and A^T*X for narrow panels, used by the norm estimator |
| `Triangular.cs` | Unblocked TRSM, blocked by 4 right-hand sides, plus transposed solves |
| `Lu.cs` | Blocked right-looking LU with partial pivoting, solve and transpose solve |
| `LinearOperators.cs` | `ILinearOperator`, dense/matrix-power and LU-inverse operators |
| `NormEstimate.cs` | `normest1`, exact 1- and inf-norms, `dgecon`-equivalent rcond |
| `LuTest.cs` | LU residual verification + block-size sweep |
| `NormEstimateTest.cs` | Estimator accuracy by ensemble, rcond checks, cost report |
| `tests/GemmLab.Tests/` | xunit suite, 649 tests, run per supported micro-kernel |
| `.github/workflows/ci.yml` | Build, test, harness, and the kernel spill gate |
| `disasm.sh` | Per-kernel JIT disassembly + accumulator-spill check, CI-gating |

The architecture is three layers, deliberately: storage/views, allocation-free
span-based kernels, then an ergonomic layer. Only the middle layer exists so
far.

---

# Measured results

Verification machine: Intel i7-12700H (6 P-cores + 8 E-cores, 20 threads,
1.25 MiB L2 per P-core, 2 MiB per 4-E-core cluster, 24 MiB L3, 45 W),
Pop!_OS 24.04, .NET 10.0.12, BLIS 0.9.0 reporting `haswell` with an optimized
native packed micro-kernel.

## Single-core GEMM: parity with BLIS

Pinned with `taskset -c 0`, medians over 31 samples:

| n | C# AVX2 8x6 | BLIS 1-thread | ratio |
| --- | --- | --- | --- |
| 256 | 51.95 | 55.81 | 93% |
| 512 | 51.44 | 53.35 | 96% |
| 1024 | 53.76 | 53.97 | 99.6% |
| 2048 | 54.32 | 54.94 | 99% |

Kernel ceiling (L1-resident, no packing) 61.63 GFLOP/s; both implementations
converge to ~88% of it, so the residual 12% is packing and memory traffic
inherent to blocked GEMM, not codegen quality.

**Stated conservatively: parity within about +/-10%.** Run-to-run drift on the
kernel ceiling alone was 6.6% (62.36 / 52.39 / 51.01 across runs), which is
larger than most individual differences in the table.

Note BLIS has no Alder Lake sub-configuration and falls back to its `haswell`
config, whose double micro-kernel is 6x8 AVX2 assembly — the same geometry and
ISA. So this is the same algorithm compiled two ways, which is exactly the
comparison that was wanted.

## Threading: power-limited, not algorithm-limited

Best results at n=2048, each measured within its own run:

| Config | 1 thread | best | threads | speedup | per-thread efficiency |
| --- | --- | --- | --- | --- | --- |
| E-cores only (8) | 19.6 | 92.2 | 6 | 4.7x | 78% |
| P-cores only (6) | 55.7 | 136.0 | 6 | 2.4x | 41% |
| P + SMT (12) | 50.3 | 145.0 | 8 | 2.9x | 36% |
| All 20 | 54.9 | 173.9 | 6 | 3.2x | — |

**The E-core result is the important one.** 4.7x on 6 threads at 78%
per-thread efficiency proves the parallel structure scales when the hardware is
not power-constrained. Same code, same block sizes, same barriers as the
P-cores managing 41%.

Evidence the limit is package power, not software:
- P-only 6 threads = 22.7 GFLOP/s per core, implying ~1.42 GHz effective
  against ~3.9 GHz implied by the single-core ceiling.
- P alone (136) + E alone (92) = 228, but together only 174. The parts do not
  add.
- Within one run, n=512 gives 160.3 but n=2048 gives 136.0 — larger problems
  should be more efficient; the difference is sustained-load duration.

Ruled out: memory bandwidth (packed-B re-streaming is ~3.8 GB/s at n=2048, DRAM
under 1 GB/s) and fork/join overhead (136 `Parallel.For` cycles at a generous
40 us each is 5.4 ms of 126 ms).

**Conclusions: SMT is worthless here** (P-only 136.0 vs P+SMT 128.7 — FMA-bound
siblings contend for the same units). E-cores add roughly 12%. Sweet spot is
6-8 threads, flat beyond. ~150-175 GFLOP/s is this machine, not this code.
A desktop part with real power headroom would tell a very different story.

## LU

Verified on the verification container (single virtualised AVX2 core), **not
yet run on the 12700H**:

| n | best nb | GFLOP/s | % of same-size GEMM |
| --- | --- | --- | --- |
| 512 | 32 | 18.9 | 49% |
| 1024 | 32 | 24.5 | 60% |
| 2048 | 64 | 26.7 | 65% |

LAPACK norm is roughly 70-80% of GEMM, so ~20% remains. Note the block-size
optimum shifts with n (32 below 2048, 64 at 2048); a size-dependent default is
probably right once measured on real hardware.

**These numbers predate routing the trailing update through `ParallelGemm`, and
the "% of same-size GEMM" column they came from compared serial LU against
serial GEMM.** Both sides now go through `GemmDispatch`, so the ratio compares
like with like and the whole table needs re-taking. Comparing a threaded LU
against a serial GEMM reads as ~97% at n=2048, which measures the thread count
and not the factorization.

Phase breakdown at n=2048, nb=64 after optimisation: GEMM 65%, panel 14%,
swaps 10%, TRSM 11%.

Residuals: `||PA-LU||_F / ||A||_F` worst 1.80e-15, `||Ax-b||_inf /
(||A||_inf ||x||_inf)` worst 3.63e-16, across 11 shapes (square, tall, wide,
straddling block boundaries), two block sizes, padded and unpadded strides,
well- and ill-conditioned inputs.

---

# Findings worth not rediscovering

1. **RyuJIT compiles a BLIS micro-kernel at parity with hand-written assembly.**
   The AVX-512 k-loop emits exactly 16 `vfmadd231pd`, 8 `vbroadcastsd` with
   folded memory operands, 2 `vmovups`, and 4 scalar loop-overhead
   instructions. Zero stack traffic; all 16 accumulators stay in zmm1-zmm16.
   Confirmed on two microarchitectures and two runtime versions.

2. **RyuJIT's register allocator is shape-sensitive and fails silently.** A flat
   loop with 16 independent FMA chains and a loop-invariant constant — same
   register count, same ISA — spilled every accumulator to the stack, ~5x off
   peak, with no warning. Disassembly checking belongs in CI. Spills were
   `rbp`-relative, not `rsp`-relative; grep for both.

3. **Tiered compilation will lie to you in benchmarks.** Large GEMMs make too
   few kernel calls to tier up inside the measurement window; before
   `AggressiveOptimization` annotations the first run reported 14 GFLOP/s at
   n=128-1024 and 73 at n=2048. Cross-check with `DOTNET_TieredCompilation=0`.

4. **Row interchanges were 21% of LU runtime.** The obvious loop order (pivots
   outer, columns inner) re-streams the whole column range once per
   interchange, touching 8 useful bytes of every 64-byte line. Transposing to
   column-outer took it to 9% and LU overall from 253 to 222 ms.

5. **Short axpys don't amortise.** TRSM ran at ~4 GFLOP/s because its inner
   axpy averages nb/2 ~ 32 elements and L was re-read per column. Blocking by
   four right-hand sides took it from 15% to 11%.

6. **`Environment.ProcessorCount` respects the affinity mask**, so
   `taskset -c 0` makes a thread-count sweep degenerate. Pin for single-thread
   comparisons; do not pin for scaling sweeps.

7. **Sequential thread-count sweeps confound thermal state with thread count**
   on a laptop part. Later configurations run heat-soaked. Re-running the sweep
   in reverse order is the cheap decisive test; this has NOT been done yet.

8. **A heuristic estimator needs invariant tests, not accuracy tests.**
   `normest1` returns a lower bound, so underestimating is correct behaviour
   and "wrong answer" and "unlucky answer" are indistinguishable from a single
   result. What pins it down is three properties that must hold exactly:
   - it never exceeds the true 1-norm, because every probe is a genuine
     `||Ax||_1` at `||x||_1 = 1`;
   - with `t = n` probe columns it is exact, because the second iteration
     probes every unit vector and therefore every column;
   - with `A >= 0` it is exact, because the first sign matrix is all ones, so
     `A^T S` is literally the vector of column sums and the first sort lands on
     the true maximiser.

   Between them these exercise the sign matrix, the transposed product, the row
   maxima, the descending sort and the unit-vector selection, without asserting
   any particular estimate. On uniform random signed matrices the estimator is
   exact only ~38% of the time at `t=2` (55% at `t=4`) — that is the ensemble,
   not a defect: random column 1-norms cluster within a few percent of each
   other, so picking the exact argmax among n near-ties is hard and missing it
   is nearly free. Worst observed ratio is 0.76, well inside the factor-of-2
   the algorithm promises.

9. **Dumping JIT disassembly through `dotnet run` loses methods.** The SDK
   driver and the application are separate processes and both honour
   `DOTNET_JitStdOutFile`, so they clobber the same file: the dump ends up with
   framework methods in it and, worse, silently missing kernels — the AVX-512
   kernel vanished from a dump that still looked plausible. Run the built
   binary directly, and scope the spill grep to the kernel listings, since
   plenty of unrelated framework methods are also called `Execute`.

10. **Code with no caller is not verified by anything.** `Lu.cs`,
    `LuTest.cs`, `Blas1.cs` and `Triangular.cs` all arrived in one commit that
    did not touch `Program.cs`, so `LuTest.Run` was unreachable and LU had
    never been exercised by this repository's entry point — while this file
    quoted its residuals as established. The residuals were real, but they came
    from somewhere else. This is the specific failure that "verification means
    a human reads stdout" produces, and the reason `--verify-only` now returns
    an exit code and CI runs it.

---

# Design decisions and why

- **Parallelism over loop 2 (jr), not loop 3 (ic).** Loop 3 gives only
  ceil(m/MC) = 15 work items at m=2048, fewer than the machine has threads, so
  the schedule would be gated by whichever block landed on an E-core. Shrinking
  MC to compensate multiplies L3 traffic for B. Loop 2 gives ~341 items of
  ~10 us each, which .NET's dynamic partitioner balances across P and E cores
  by itself.
- **No P/E core detection.** An earlier plan for a startup calibration pass was
  dropped: dynamic work distribution achieves the same balance with none of the
  machinery, is self-correcting under thermal throttling, and needs no
  Linux-specific `/sys` reads.
- **A is packed for the whole m x KC slab in one parallel phase**, not per-MC
  block, which removes a barrier per block without costing locality — the
  compute phase still walks MC blocks in order, so all threads read the same A
  block simultaneously.
- **Block sizes derived from cache geometry**: KC=256 so one A plus one B
  micro-panel fit a Gracemont E-core's 32 KiB L1 (48 KiB on Golden Cove);
  MC=144 so the packed A block fits an E-core's quarter-share of its cluster's
  2 MiB L2. Sized for the E-cores deliberately — a single slow worker gates the
  schedule.
- **LU verified by residual, never by element-wise comparison.** On ties the
  pivot choice is arbitrary, so two correct implementations legitimately
  produce different L, U and P from the same input.
- **Exact vs numerical singularity are different things.** A duplicated column
  is mathematically singular but its pivot is rounding noise, not exact zero,
  so factorization completes — identical to `dgetrf`. `PivotRatio` is a cheap
  indicator, explicitly NOT a condition number.

---

# Open items

- **`Gemm.cs` still uses placeholder MC=288, KC=384.** The cache-derived
  KC=256/MC=144 used in `ParallelGemm` measured better on the 12700H at every
  size below 2048 (n=128: 47.0 vs 35.1 GFLOP/s). This has not been ported to
  the serial path.
- **Small-n threading.** A work-based threshold *is* applied in
  `ParallelGemm.Multiply` —
  `Math.Clamp((int)(work / 8_000_000), 1, scratch.MaxThreads)` — with divisor
  8M rather than the 12M once drafted, and no cap at 8: `MaxThreads` still
  defaults to `ProcessorCount`. Whether this actually fixed the n=128 case has
  not been re-measured on the 12700H, so the earlier 0.8-1.0x figures may
  predate it.
- **`GemmDispatch.ParallelThreshold` (4e6 flops) is derived, not measured.**
  It decides serial vs threaded for every LU trailing update. Sweep it.
- **LU has not been run on the 12700H**, and its results table now needs
  re-taking anyway (see above).
- **Reverse-order thread sweep not done** (see finding 7).
- **`normest1` has not been cross-validated against MATLAB's `normest1` or
  LAPACK's `dlacn2`.** It is verified by invariants instead — see finding 8 —
  which is strong evidence but not the same thing.
- **The estimator's `Blas2` products are O(n^2 t) with no blocking.** Fine at
  the sizes that matter for `dgecon`, possibly not for `expm`'s inner loop.

# Next steps, in priority order

`normest1` and the LU/`ParallelGemm` routing that used to head this list are
done; so are the test project, CI and the licence. What remains:

1. **`expm`** via Al-Mohy & Higham (2009) scaling-and-squaring with degree-13
   Pade. Use the 2009 algorithm, not Higham 2005: it picks the scaling from
   estimates of `||A^k||^(1/k)` rather than `||A||`, which specifically
   mitigates overscaling on stiff matrices — exactly the MTL/FEM state matrices
   this is for. Moler & Van Loan's "Nineteen Dubious Ways" is the reference to
   keep open.
2. **`expmv`** (Al-Mohy & Higham 2011) — computes `exp(A t) b` without forming
   the exponential, a few dozen matvecs instead of ~15-25 GEMM-equivalents.
   For a transient sweep this is likely a 100x algorithmic win that dwarfs any
   further kernel tuning, and it is the natural bridge to sparse.
3. Re-take the LU table on the 12700H now that both sides of the ratio use the
   same GEMM path, and sweep `GemmDispatch.ParallelThreshold` while there.
4. Recursive (Toledo) panel factorization to push LU from 65% toward 75-80% of
   GEMM.
5. Complex support. `System.Numerics.Complex` is interleaved, which matches
   `zgemm` layout but vectorises badly; a split (SoA) representation is 2-4x
   faster for element-wise work. Start interleaved, switch only if profiling of
   real assembly workloads says so.

## Deliberately deferred

SVD, nonsymmetric eigenvalues, Schur. These are where "roll your own" is
genuinely unwise; port from reference LAPACK if and when needed.

## Longer-term design headroom

The argument that a 2026 C# library can beat the incumbents rests less on FLOPs
than on things no BLAS-lineage library can express:

- **Structure in the type system.** `Matrix<T, TStructure>` with static
  abstract dispatch so `A.Solve(b)` picks Cholesky vs LU vs triangular
  substitution at compile time. LAPACK encodes this in function names
  (`dposv` vs `dgesv`); getting it wrong is silent. This eliminates a real bug
  class at zero runtime cost.
- **One algorithm, every numeric type.** LAPACK maintains four hand-written
  copies (s/d/c/z) that drift. Generic math means one LU instantiated for
  `float`, `double`, `Complex`, `Half`, `BFloat16` — and double-double or
  interval arithmetic for rigorous error bounds, which no .NET library offers.
- **Mixed precision as composable types** rather than bolted-on routines.
- **Matrix functions as first-class**: `expm`, `logm`, `sqrtm`, phi-functions,
  `f(A)b`. LAPACK essentially lacks these; .NET has nothing good.
- **Reproducibility mode** with deterministic reduction order. Multi-threaded
  BLAS gives run-to-run variation; bit-reproducible results matter for
  engineering reports and regression tests.

---

# Testing

```
dotnet test -c Release                  # unit suite
dotnet run -c Release -- --verify-only  # whole-program harness, exits non-zero on failure
./disasm.sh                             # kernel codegen gate
```

Three layers, deliberately overlapping:

- **The xunit suite** (`tests/GemmLab.Tests`) is the regression net. Every
  contract that is generic over the micro-kernel runs once per kernel, via an
  abstract base class with one concrete subclass each; a kernel the host cannot
  run is reported *skipped*, never silently passed. Internals are visible to it
  because `Packing`, `Blas1`, `Blas2` and `Triangular` are exactly where an
  off-by-one hides.
- **The console harness** (`Program.cs`) covers what unit tests cannot: BLIS
  dispatch reporting, the benchmark plumbing itself, and sweeps large enough to
  be worth reporting rather than asserting. `--verify-only` skips the
  benchmarks and returns an exit code.
- **`disasm.sh`** is the codegen gate, and it is the one CI job that cannot be
  replaced by a test: correctness is unaffected by a spill, only speed is.

Guidance that has already been paid for once:

- **Never assert bit-identical results between two code paths.** The blocked and
  unblocked LU paths agree to 7e-15, not to zero, because the blocked one sums
  its trailing update through GEMM. Assert the pivot sequence exactly (integer
  choices) and the factors by tolerance.
- **Test a heuristic by its invariants**, not by its accuracy. See finding 8.
- **A `Skip` that is really an early `return` is a lie.** This is why the suite
  is on xunit.v3, which has `Assert.Skip`.

---

# Benchmarking methodology

Non-negotiable, because several early conclusions were artifacts:

- Medians and IQR over 31 samples, never best-of-N. GFLOP/s from median
  latency.
- `[MethodImpl(MethodImplOptions.AggressiveOptimization)]` on every hot path.
- `taskset -c 0` for single-thread comparisons; unpinned for scaling.
- Check the disassembly for spills after any kernel change: run `./disasm.sh`,
  which does this per kernel and exits non-zero on a spill. CI runs it too.
  Do NOT dump the disassembly via `dotnet run`: the SDK and the app are
  separate processes and both honour `DOTNET_JitStdOutFile`, so they overwrite
  each other's output and kernels silently go missing from the dump. Run the
  built binary directly.
- Verify BLIS dispatch via `bli_arch_string` before quoting any ratio; a
  `generic` architecture means the comparison is against a fallback.
- Capture `turbostat --interval 1` alongside multi-threaded runs on laptop
  parts so throttling is separable from scheduling.
