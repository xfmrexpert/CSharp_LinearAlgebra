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

A library plus tests, benchmarks and a diagnostics tool, `net10.0`,
column-major throughout, unit row stride. `Matrix<T>` is generic over
`unmanaged, INumberBase<T>`, so storage, views and the structure vocabulary
already work for any numeric type; arithmetic is double-only and lives in
extensions on the closed `Matrix<double>`, so adding a type is additive.

Storage is a GC-pinned managed array: nothing in the core is `IDisposable`, a
live view keeps its storage alive, and use-after-free is unexpressible. Views
carry a `Span<T>`, so the runtime is the last line of defence on every slice.
The public assembly compiles with `AllowUnsafeBlocks=false` and
`CheckForOverflowUnderflow=true`; every pointer lives in a second assembly,
`Tensile.Kernels`, whose types are all `internal`. The BLIS binding is a
third assembly and a separate package that references the core, never the
reverse. Every allocation sized by a request goes through `Storage`, which
refuses anything over `TensileLimits.MaxElements` before asking the runtime.
See `docs/security-design.md`; Phases 1–6 of it have landed and the invariant
suite is green. CI pins every action to a commit SHA, restores in locked mode
against committed `packages.lock.json`, runs CodeQL, and fuzzes nightly.

The three-layer architecture this file has described from the start now
exists in full, as three assemblies:

| Layer | Assembly / namespace | Holds |
| --- | --- | --- |
| Ergonomic | `Tensile` (public, no unsafe) | `Matrix<T>`, `MatrixView<T>`, structures, `LuDecomposition`, `Workspace`, the fluent operations, `ILinearOperator`, `NormEstimate` |
| Kernels | `Tensile.Kernels` (all internal, unsafe) | Micro-kernels, packing, the GEMM drivers, LU, triangular solves, Blas1/2, exact norms, the `KernelEntry` seam |
| Interop | `Tensile.Interop.Blis` (separate package, public over views) | The optional BLIS binding, for benchmarks; the only native loading anywhere |

The kernel assembly is reached only through `KernelEntry`, which takes
`Operand`/`Target` (a span plus rows, columns, stride) and pins with `fixed`
for exactly one call. Nothing it returns holds a pointer: `LuFactorization`
carries pivots and diagnostics only, and the solves take the factors back as an
operand. The project's own tests, benchmarks and diagnostics see the internals
through `InternalsVisibleTo`; a consumer of the package cannot.

| Path | Purpose |
| --- | --- |
| `src/Tensile/Matrix.cs` | Owning storage on the pinned object heap, 64-byte aligned, not disposable; plus the creation factories |
| `src/Tensile/MatrixShape.cs` | Self-validating shape value; all extent/offset arithmetic, checked in `long` |
| `src/Tensile/MatrixView.cs` | `MatrixView<T>` / `ReadOnlyMatrixView<T>`: `Span`-backed ref structs, obtained only by `Bind` or slicing; no pointer constructor |
| `src/Tensile/ViewOperands.cs` | Repackages a view as a kernel `Operand`/`Target`; the only place the public assembly touches the kernel assembly's types |
| `src/Tensile/LinearOperators.cs` | `ILinearOperator` over views (the public extension point), `DenseMatrixOperator` (A^p), `LuInverseOperator` |
| `src/Tensile/NormEstimate.cs` | Higham–Tisseur `normest1` as safe code over managed arrays; `Condition` (dgecon-equivalent) |
| `src/Tensile/TensileLimits.cs` | `TensileLimits.MaxElements` (process-wide ceiling), `AllocationLimitException`, and `Storage` — the one allocation path every request-sized buffer goes through |
| `src/Tensile/Structures.cs` | `IMatrixStructure`, `ITriangularStructure`, General + the three triangular structures, `StructuredMatrix<T, TStructure>` |
| `src/Tensile/Workspace.cs` | Kernel choice + packing buffers, internally locked |
| `src/Tensile/LuDecomposition.cs` | Owning factorization; captures ||A||_1 before overwriting A |
| `src/Tensile/MatrixOperations.cs` | Fluent extensions on `Matrix<double>` and the structured solves |
| `src/Tensile.Kernels/KernelEntry.cs` | The single seam where spans are pinned and become pointers; restates every shape precondition |
| `src/Tensile.Kernels/Operand.cs` | `Operand` / `Target`: span + shape, length-checked on construction |
| `src/Tensile.Kernels/Alignment.cs` | Cache-line offset for pinned arrays; the one address read on the allocation path |
| `src/Tensile.Kernels/*.cs` | As before: kernels, packing, Gemm/ParallelGemm/GemmDispatch, Blas1/2, Triangular, Lu, Norms, Reference |
| `src/Tensile.Interop.Blis/` | Native `bli_dgemm` binding + dispatch/ABI queries, its own package; `README.md` carries the `TENSILE_BLIS_LIBRARY` warning |
| `tests/Tensile.Fuzz/` | SharpFuzz harness: an input is a script of operations over hostile integers; the property is I5. Nightly under afl++; `--self-check` replays the seed corpus per PR |
| `tests/Tensile.Tests/` | xunit.v3, 830 tests; `Invariants/` is the secure-by-design spec, all green; kernel-generic contracts run per kernel via `IKernelCase` markers |
| `bench/Tensile.Benchmarks/` | BenchmarkDotNet: GEMM vs BLIS, kernel ceiling, LU block-size sweep, API overhead (what the security migration cost), thread scaling, serial/threaded crossover, serial cache-blocking sweep |
| `tools/Tensile.Diagnostics/` | `tensile-diag`: ISA, BLIS dispatch, estimator accuracy; and the codegen gate's process |
| `disasm.sh` | Per-kernel disassembly + accumulator-spill check |
| `docs/api.md` | The API guide |

`GenerateDocumentationFile` with `TreatWarningsAsErrors` makes an undocumented
public member a build error. That is the mechanism keeping the API documented;
it is not optional politeness.

---

# Measured results

Verification machine: Intel i7-12700H (6 P-cores + 8 E-cores, 20 threads,
1.25 MiB L2 per P-core, 2 MiB per 4-E-core cluster, 24 MiB L3, 45 W),
Pop!_OS 24.04, .NET 10.0.12, BLIS 0.9.0 reporting `haswell` with an optimized
native packed micro-kernel.

## Single-core GEMM: parity with BLIS

Re-taken 2026-09-19 under BenchmarkDotNet, `taskset -c 0`, 31 iterations,
GFLOP/s from the mean (medians agree within 1% at every size; BDN reported a
median column for the managed side only). Within-run dispersion was 1.2-2.0%
StdDev on both sides. AVX-512 is fused off on this part — `tensile-diag`
reports `AVX-512F: False` — so the AVX2 8x6 kernel is what runs.

| n | C# AVX2 8x6 | BLIS 1-thread | ratio | C# % of ceiling | BLIS % of ceiling |
| --- | --- | --- | --- | --- | --- |
| 128 | 49.57 | 58.40 | 85% | 79% | 93% |
| 256 | 54.27 | 55.81 | 97% | 86% | 88% |
| 512 | 56.31 | 57.72 | 98% | 89% | 92% |
| 1024 | 57.26 | 55.85 | **103%** | 91% | 89% |
| 2048 | 57.22 | 57.39 | 99.7% | 91% | 91% |

Kernel ceiling (L1-resident, no packing, AVX2 8x6) 63.02 GFLOP/s; the scalar
4x4 kernel manages 11.05. Both GEMM implementations converge to ~91% of the
ceiling, so the residual ~9% is packing and memory traffic inherent to blocked
GEMM, not codegen quality. For scale, one Golden Cove core at ~4.0 GHz has a
64 GFLOP/s AVX2 FMA peak, so the micro-kernel is running at ~98% of what the
hardware can issue.

**The managed GEMM is at parity from n=256 up, and 3% ahead at n=1024.**
That is the headline result of the whole project: the same algorithm, written
in C# and compiled by RyuJIT, against hand-written assembly kernels in a
mature native library, on the same core.

Treat the ratio and not either column as the finding. Both sides came out
faster than the previous recording (C# 51.9-54.3 -> 54.3-57.3, BLIS
53.4-55.8 -> 55.8-57.7) on the same machine and the same BLIS build, which
says cross-run conditions moved; within-run dispersion of 1-2% is much smaller
than that drift. Note also that `GemmBenchmarks` allocates with `NativeMemory`
and calls the dispatch directly, so none of this touches the POH storage or
the validation seam — the improvement is not attributable to the
secure-by-design work, and `ApiOverheadBenchmarks` is what measures that.

**n=128 is the one place BLIS is clearly ahead** (85%), and it is also the
size where the managed side is furthest from its own ceiling (79% against
BLIS's 93%). The obvious suspect was the serial path's placeholder MC=288,
KC=384, both larger than the entire 128x128 operand — **and the sweep below
ruled it out.** At n=128 all four block-size combinations land within 1% of
each other. Whatever costs the managed side 15% at n=128, it is not cache
blocking. Per-call overhead that O(n^3) has not yet amortised is the next
hypothesis, and `ApiOverheadBenchmarks` measures a related quantity.

Note this table was taken with the old MC=288; the serial default is now
MC=144, so it needs a quick re-run to stay current.

Note BLIS has no Alder Lake sub-configuration and falls back to its `haswell`
config, whose double micro-kernel is 6x8 AVX2 assembly — the same geometry and
ISA. So this is the same algorithm compiled two ways, which is exactly the
comparison that was wanted.

The `threaded` rows in that run are not a threading result and are not
recorded: pinning to one core makes `ProcessorCount` 1, so they measure the
parallel path's fork/join overhead against a single worker. That overhead is
worth one number — 1-3% at n>=256, and exactly 0 at n=128, where
`work = 2.1e6` falls below `ParallelThreshold` and the dispatch takes the
serial path outright (confirmed by the threaded row allocating nothing at all
at that size). Real scaling comes from `ThreadScalingBenchmarks`, unpinned.

## Serial cache blocking: MC measured, KC is noise

Swept on the 12700H, pinned, 31 iterations, run twice — ascending and
descending (`TENSILE_BENCH_REVERSE=1`) — because a one-directional sweep
cannot tell a block size from a cooler. GFLOP/s from medians.

MC=144 against the old placeholder MC=288, best KC for each:

| n | ascending | descending | verdict |
| --- | --- | --- | --- |
| 128 | MC144 +13.8% | MC144 -0.7% | sign flips: thermal |
| 512 | MC144 +12.3% | MC144 -1.5% | sign flips: thermal |
| 2048 | MC144 +9.6% | MC144 **+9.5%** | **survives: real** |

**Only n=2048 survives reversal, and it survives to a tenth of a percent.**
MC=144 is now the serial default, matching what `ParallelGemm` already
derived from cache geometry. KC=256 and KC=384 came out within 1% of each
other everywhere in both directions, so KC stays at 384 — there is nothing to
choose between them, and moving it would be unmeasured churn.

The ascending-only run would have reported "MC=144 is 10-14% better at every
size". Two of those three numbers were the machine warming up. This is the
first time the reverse-order test of finding 7 has actually been run, and it
overturned two results out of three.

One thing this sweep did NOT explain: `BlockSizeBenchmarks` at MC=288/KC=384
is bit-for-bit the same configuration and driver as `GemmBenchmarks`'s serial
row, yet ran 8-18% slower in every attempt, including the clean
high-priority one. Not tiering — `Gemm.Multiply`, `ParallelGemm.Multiply` and
both `GemmDispatch` entry points all carry `AggressiveOptimization`, so that
was checked and ruled out. Most likely the single-core GEMM table was taken
on the coldest machine of the session. It does not affect the MC conclusion,
which is a within-run comparison replicated in both directions, but it does
mean absolute GFLOP/s are not comparable across benchmark classes in one
sitting.

## Threading: power-limited, and the baseline is a trap

Re-taken on the 12700H, 2026-09-19: unpinned (finding 6), root so BDN could
raise priority, 31 iterations, `ParallelGemm` driven directly so the dispatch
threshold cannot divert a row. GFLOP/s from medians.

| threads | n=512 | n=2048 |
| --- | --- | --- |
| 1 | 48.21 | 44.32 |
| 2 | 87.72 | 86.49 |
| 4 | 118.62 | 133.98 |
| 6 | 133.42 | **170.65** |
| 8 | **150.05** | 151.33 |
| 12 | 136.61 | 145.81 |
| 16 | 128.62 | 139.72 |
| 20 | 121.79 | 142.25 |

Peak 170.65 GFLOP/s at 6 threads (n=2048) and 150.05 at 8 (n=512), then a
17-19% decline out to 20 threads. The peak location and magnitude replicate
the earlier campaign (which also peaked at 6 threads, at 173.9), so
**~150-175 GFLOP/s at 6-8 threads is this machine, twice measured.**

**The 1-thread row is not a valid baseline and speedups quoted against it are
inflated.** Unpinned, a `Parallel.For` with `MaxDegreeOfParallelism = 1` is
scheduled wherever the OS likes, including onto a Gracemont E-core — and the
n=2048 1-thread row has the worst dispersion in the table at 6.6% StdDev,
which is what that drift looks like. It reads 44.32 GFLOP/s against the
57.22 measured pinned on a P-core, so it understates the baseline by 29% and
overstates every speedup by the same factor. Against the pinned figure the
real peak speedup is 2.98x on 6 threads (50% per-thread efficiency), not the
3.85x the in-run baseline suggests.

This is the sharp edge of finding 6: you must not pin a scaling sweep, and
the 1-thread row of an unpinned sweep is therefore worthless as a
denominator. Take the baseline from a separate pinned run.

*The decline beyond the peak is NOT yet confirmed.* Thread count and
execution order are the same variable in this sweep — higher thread counts
run both later and hotter — which is exactly the confound of finding 7. A
descending run is needed before the shape past the peak can be trusted. The
peak location is safe (it reproduces across two campaigns); the tail is not.

## LU

Re-taken on the 12700H, 2026-09-19: threaded (the trailing update goes through
`GemmDispatch`, so sizes above `ParallelThreshold` use every core), unpinned,
31 iterations, GFLOP/s from medians at (2/3)n^3 - n^2/2 - n/6 flops.

| n | nb=32 | nb=64 | nb=128 |
| --- | --- | --- | --- |
| 256 | **20.16** | 17.54 | 13.49 |
| 512 | **30.71** | 28.56 | 23.11 |
| 1024 | 42.68 | **44.57** | 35.91 |
| 2048 | 64.96 | **65.52** | 52.71 |

The old container table (18.9 / 24.5 / 26.7 at 512 / 1024 / 2048, single
virtualised AVX2 core) is superseded and not comparable — different machine,
different thread count.

**The block-size ranking is only half settled, and the ordering says which
half.** BenchmarkDotNet runs nb=32 before 64 before 128 within each size, so
later block sizes run hotter (finding 7):

- **nb=64 at n>=1024 is credible**: it wins *despite* running second, into a
  thermal headwind. An effect that survives a headwind is worth more than one
  that rides a tailwind.
- **nb=32 at n<=512 is not established**: it wins while running first, which
  is the exact shape the serial block-size sweep produced and then reversed.
- **nb=128 losing by 19-33% is mostly real**, because there is a mechanism
  rather than just an ordering: the panel factorization is O(m*nb^2) level-2
  work, so a wider panel moves more of the total into the slow unblocked
  path. That deficit is also far larger than the ~12-14% thermal artifact
  measured in the serial sweep.

A descending run (`TENSILE_BENCH_REVERSE=1`) would settle the small-n
crossover. Until then the honest statement is: nb=64 above n=1024, and the
optimum below that is unknown.

**Against threaded GEMM at the same thread count, LU is at 46%** — and the
arithmetic explains the whole gap without implicating the factorization.

| n | LU (best nb) | threaded GEMM, 20 threads | ratio |
| --- | --- | --- | --- |
| 512 | 30.71 | 121.79 | 25% |
| 2048 | 65.52 | 142.25 | **46%** |

LAPACK's norm is 70-80%, so 46% looks alarming. It is not a defect, it is
Amdahl, and the numbers close:

- Threaded GEMM is 2.49x single-core GEMM (142.25 / 57.22).
- Apply that to the GEMM share alone of the phase breakdown below — 65% GEMM,
  35% panel + swaps + TRSM, all three of which are serial — and the
  factorization should speed up by 100 / (65/2.49 + 35) = **1.64x**.
- Working backwards from the observed 65.52 threaded, serial LU would be
  **39.95 GFLOP/s, or 69.8% of serial GEMM** — squarely inside LAPACK's band.

So the factorization itself is healthy at roughly 70% of GEMM, exactly as it
should be, and the threaded ratio collapses only because barely two thirds of
the work is threaded at all. **The bottleneck is the serial 35%, not the
blocked algorithm.** That is a far sharper case for recursive (Toledo) panel
factorization than "LU is at 65% of GEMM" ever was: shrinking the panel
attacks the 14%, and threading the swaps and TRSM attacks the other 21%.

Two caveats on that reconciliation. The phase breakdown it leans on was
measured single-core on the old container, so the split on this machine may
differ; and LU and the GEMM sweep are different benchmark classes, which have
been seen to disagree by 8-18% in absolute terms within one sitting. The
conclusion is robust to both — a 46% ratio would have to be wrong by a very
large factor to reach 70% — but the second decimal place is not.

Phase breakdown at n=2048, nb=64 — GEMM 65%, panel 14%, swaps 10%, TRSM 11% —
is from the old single-core container run and has NOT been re-taken threaded.
Re-taking it is now the highest-value LU measurement, because it is the input
to the Amdahl argument above.

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

   The corollary bites on a hybrid part: **the 1-thread row of an unpinned
   sweep is worthless as a baseline.** A `Parallel.For` limited to one worker
   still goes wherever the scheduler puts it, and on a 12700H that includes
   the E-cores. Measured, the unpinned 1-thread row read 44.32 GFLOP/s against
   57.22 pinned to a P-core — understating the baseline by 29% and inflating
   every speedup in the table by the same factor (3.85x claimed against 2.98x
   real). It also had the worst dispersion in the sweep, 6.6% StdDev, which is
   the tell. Take the numerator from the unpinned sweep and the denominator
   from a separate pinned run.

7. **A one-directional sweep cannot tell a parameter from a cooler.** On a
   laptop part, later configurations run heat-soaked, so any sweep confounds
   its parameter with thermal state. Re-running in reverse order is the cheap
   decisive test, and it is no longer hypothetical: the serial block-size
   sweep reported MC=144 ahead by 13.8% at n=128, 12.3% at n=512 and 9.6% at
   n=2048 going up, and by -0.7%, -1.5% and +9.5% coming down. **Two of the
   three results were the machine warming up.** Only the n=2048 effect was
   real, and it reproduced to a tenth of a percent.

   The mechanism is worth stating because it is not obvious: BenchmarkDotNet
   sorts cases by parameter value before executing them, so the whole first
   half of a two-value sweep runs on a cooler machine than the second half.
   Reversing a `ParamsSource` list does not help — BDN re-sorts it. The
   reversal has to be a custom `IOrderer` overriding `GetExecutionOrder`
   (`ThermalOrderer`, driven by `TENSILE_BENCH_REVERSE=1`), which was itself
   verified by diffing the `// Benchmark:` lines of two runs rather than
   assumed to work.

   Running as root so BDN can raise process priority tightened dispersion
   from 1.4-8.4% StdDev to roughly 1%, and is worth doing for any sweep whose
   effect size is in single-digit percent.

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
   any particular estimate.

   Exactness depends heavily on the ensemble, which is why `tensile-diag`
   reports several rather than one number. On *uniform* random signed matrices
   it is exact 26% of the time at `t=2` and 41% at `t=4`; with a few dominant
   columns (the earlier ensemble, and the more realistic one) it is 61% and
   79%. Neither is a defect: uniform column 1-norms cluster within a few
   percent, so picking the exact argmax among n near-ties is hard and missing
   it is nearly free. Worst observed ratio is 0.76, well inside the factor of
   two the algorithm promises. An earlier revision of this file quoted 38% and
   55% from the skewed ensemble alone.

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

11. **An empty matrix can still have two billion columns.** `MatrixShape(0, n)`
    is valid for any `n` up to `int.MaxValue` — its extent is zero — and every
    loop of the form `for j < Columns` walked all of them to touch nothing.
    `Matrix.FromColumnMajor<double>(0, 1_546_977_280, [])` took 36 seconds
    from a 16-byte input. The property tests never caught it because they
    enumerate hostile values one argument at a time and zero-with-huge is a
    pair; the fuzzer found it within 90 seconds of its first run. Every
    column walk in the public assembly and every kernel entry point now
    returns first on an empty operand, and `EmptyShapeTests` pins it with a
    generous time bound. The general lesson: validity and cost are different
    questions, and an input that is valid can still be a denial of service.

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

- ~~**`Gemm.cs` uses placeholder MC=288, KC=384.**~~ *Closed.* Swept on the
  12700H in both directions; MC=144 is a real 9.5% win at n=2048 and is now
  the serial default, KC stays 384 because 256 and 384 are within 1% of each
  other everywhere. See "Serial cache blocking" above. What remains is that
  NC=4096 has never been varied at all, and that the sweep covered only three
  sizes — a size-dependent MC may still be worth having.
- **Small-n threading.** A work-based threshold *is* applied in
  `ParallelGemm.Multiply` —
  `Math.Clamp((int)(work / 8_000_000), 1, scratch.MaxThreads)` — with divisor
  8M rather than the 12M once drafted, and no cap at 8: `MaxThreads` still
  defaults to `ProcessorCount`. Whether this actually fixed the n=128 case has
  not been re-measured on the 12700H, so the earlier 0.8-1.0x figures may
  predate it.
- **`GemmDispatch.ParallelThreshold` (4e6 flops) is derived, not measured.**
  It decides serial vs threaded for every LU trailing update. Now a per-dispatch
  property rather than a `const`, defaulting to `DefaultParallelThreshold`, so
  a measurement can move it without a rebuild. `ParallelCrossoverBenchmarks`
  measures the crossover directly instead of sweeping the threshold — running
  both paths explicitly across shapes that bracket it, so the size where
  threaded first wins IS the value to set. Two shape families, because they do
  not agree: square, and the `m x m x 64` panels LU's trailing update actually
  produces. *Not yet measured on the 12700H.*
- **LU's block size is settled only above n=1024** (nb=64, credible because it
  wins against the thermal gradient). Below that, nb=32 won with the gradient
  behind it, which the serial sweep taught us not to trust; a descending run
  would settle it. And the "% of same-size GEMM" column still needs threaded
  GEMM figures at matching sizes before it can be computed at all.
- **The thread sweep's tail needs a descending run.** The ascending sweep is
  taken (see "Threading" above) and its peak is trustworthy — 6-8 threads,
  150-175 GFLOP/s, replicated across two campaigns. The 17-19% decline past
  the peak is not: thread count and execution order are the same variable, so
  higher counts run both later and hotter. `TENSILE_BENCH_REVERSE=1` settles
  it in one command.
- **The LU phase breakdown is single-core and from the old container.** It is
  now load-bearing — the Amdahl argument that explains LU's 46% of threaded
  GEMM rests on the 65/14/10/11 split — so re-taking it threaded on this
  machine is the highest-value LU measurement outstanding.
- **`normest1` has not been cross-validated against MATLAB's `normest1` or
  LAPACK's `dlacn2`.** It is verified by invariants instead — see finding 8 —
  which is strong evidence but not the same thing.
- **The estimator's `Blas2` products are O(n^2 t) with no blocking.** Fine at
  the sizes that matter for `dgecon`, possibly not for `expm`'s inner loop.
- **What the secure-by-design migration cost is still unmeasured.** Every GEMM
  benchmark but one allocates with `NativeMemory` and calls the dispatch
  directly, which skips POH storage, `Bind`, the `Operand` checks, the
  workspace lock and the pinning seam — i.e. everything the migration added.
  `ApiOverheadBenchmarks` closes that hole with three rows (native pointers /
  POH storage direct / public API) so the layers can be told apart. All of it
  is O(1) per call against O(n^3) work, so the ratio should converge to 1 as N
  grows and any real cost should show at N=128; a ratio that does not converge
  would mean per-element work, which is a defect rather than a tax. This is the
  section 9 guardrail of `docs/security-design.md`. *Not yet measured on the
  12700H.*
- **`Workspace` serialises every operation on one lock.** Correct and cheap
  against O(n^2) work, but it means concurrent independent solves on a shared
  workspace queue. Only worth revisiting if a real workload wants many small
  factorizations in parallel, where per-thread workspaces are the answer.
- **Pinned-object-heap storage is never compacted.** A hot loop creating
  thousands of tiny matrices fragments the POH. Fine for a solver holding a
  handful of large operands; a pooled allocator behind `Storage` is the fix if
  a churn-heavy workload ever appears, and is not written.
- **No `SymmetricPositiveDefinite` structure**, because there is no Cholesky to
  dispatch to. This is the missing half of the type-system argument.

# Next steps, in priority order

`normest1`, the LU/`ParallelGemm` routing, the test project, CI and the licence
are done, and so is the library shaping: `src`/`tests`/`bench`/`tools` layout,
a documented public API, and structure-typed dispatch. What remains:

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
4. **Cholesky**, which is the cheapest way to make the structure vocabulary pay
   off twice over: FEM mass and stiffness matrices are symmetric positive
   definite, and it is the third genuinely different `Solve` path.
5. Recursive (Toledo) panel factorization to push LU from 65% toward 75-80% of
   GEMM.
6. Complex support. `System.Numerics.Complex` is interleaved, which matches
   `zgemm` layout but vectorises badly; a split (SoA) representation is 2-4x
   faster for element-wise work. Start interleaved, switch only if profiling of
   real assembly workloads says so.

## Deliberately deferred

SVD, nonsymmetric eigenvalues, Schur. These are where "roll your own" is
genuinely unwise; port from reference LAPACK if and when needed.

## Longer-term design headroom

The argument that a 2026 C# library can beat the incumbents rests less on FLOPs
than on things no BLAS-lineage library can express:

- **Structure in the type system** — *done, in first form.* `IMatrixStructure`
  with static abstract dispatch, so `A.Solve(b)` picks triangular substitution
  over LU at compile time. The remaining half is Cholesky: `General` and the
  three triangular structures exist, `SymmetricPositiveDefinite` does not,
  because nothing would dispatch to it yet. Note the deliberate limit on what a
  structure claims — which triangle is *read*, not that the rest is zero —
  since packed LU makes the stronger reading impossible.
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
dotnet test Tensile.slnx -c Release                        # unit suite
dotnet run -c Release --project tools/Tensile.Diagnostics  # what this host supports
./disasm.sh                                                # kernel codegen gate
dotnet tests/Tensile.Fuzz/bin/Release/net10.0/Tensile.Fuzz.dll --self-check   # seed corpus replay
```

To fuzz locally: `apt install afl++`, `dotnet tool install -g SharpFuzz.CommandLine`,
publish `tests/Tensile.Fuzz` twice (one copy to instrument with `sharpfuzz
Tensile.dll`, one to replay findings on — instrumented code faults outside
afl), then `AFL_SKIP_BIN_CHECK=1 afl-fuzz -i bin/Corpus -o findings -t 10000
-m none -- dotnet bin/Tensile.Fuzz.dll`. Nothing from the instrumented
assembly may run before `Fuzzer.OutOfProcess.Run` attaches the coverage map.

Four layers, deliberately overlapping:

- **The xunit suite** (`tests/Tensile.Tests`) is the regression net. Every
  contract generic over the micro-kernel runs once per kernel, via an abstract
  base class with one concrete subclass each; a kernel the host cannot run is
  reported *skipped*, never silently passed. Internals are visible to it because
  `Packing`, `Blas1`, `Blas2` and `Triangular` are exactly where an off-by-one
  hides.
- **`tensile-diag`** covers what unit tests cannot: which kernels this host
  actually has, what BLIS dispatched to, and estimator accuracy by ensemble —
  numbers worth reporting rather than asserting.
- **`disasm.sh`** is the codegen gate, and the one CI job no test can replace:
  correctness is unaffected by a spill, only speed is.
- **The fuzz harness** (`tests/Tensile.Fuzz`) finds the argument *pairs* the
  property tests did not enumerate. It found finding 11 in its first 90
  seconds. A hang is a finding as much as a crash is: the property is "returns
  or throws a documented exception", and "returns after 36 seconds" fails it.

Guidance that has already been paid for once:

- **Never assert bit-identical results between two code paths.** The blocked and
  unblocked LU paths agree to 7e-15, not to zero, because the blocked one sums
  its trailing update through GEMM. Assert the pivot sequence exactly (integer
  choices) and the factors by tolerance.
- **Test a heuristic by its invariants**, not by its accuracy. See finding 8.
- **A `Skip` that is really an early `return` is a lie.** This is why the suite
  is on xunit.v3, which has `Assert.Skip`.
- **The codegen gate needs a single process.** BenchmarkDotNet spawns a child
  process per benchmark and both honour `DOTNET_JitStdOutFile`, so it cannot
  host the dump — the same failure as running it through `dotnet run`. That is
  what `tools/Tensile.Diagnostics` is for.

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
