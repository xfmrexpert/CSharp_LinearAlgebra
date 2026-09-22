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
| Ergonomic | `Tensile` (public, no unsafe) | `Matrix<T>`, `MatrixView<T>`, structures, `LuDecomposition`, `Workspace`, the fluent operations, `ILinearOperator`, `NormEstimate`, `MatrixExponential` |
| Kernels | `Tensile.Kernels` (all internal, unsafe) | Micro-kernels, packing, the GEMM drivers, LU, triangular solves, the streamed column/panel primitives, exact norms, the `KernelEntry` seam |
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
| `src/Tensile/LinearOperators.cs` | `ILinearOperator` / `ITransposableOperator` over views (the public extension points), `DenseMatrixOperator` (A^p), `LuInverseOperator` |
| `src/Tensile/NormEstimate.cs` | Higham–Tisseur `normest1` as safe code over managed arrays; `Condition` (dgecon-equivalent) |
| `src/Tensile/MatrixExponential.cs` | `Expm`, Al-Mohy & Higham (2009) scaling-and-squaring; the Padé ladder, the `ell` correction, `ExpmDiagnostics` |
| `src/Tensile/TensileLimits.cs` | `TensileLimits.MaxElements` (process-wide ceiling), `AllocationLimitException`, and `Storage` — the one allocation path every request-sized buffer goes through |
| `src/Tensile/Structures.cs` | `IMatrixStructure`, `ITriangularStructure`, General + the three triangular structures, `StructuredMatrix<T, TStructure>` |
| `src/Tensile/Workspace.cs` | Kernel choice + packing buffers, internally locked |
| `src/Tensile/LuDecomposition.cs` | Owning factorization; captures ||A||_1 before overwriting A |
| `src/Tensile/MatrixOperations.cs` | Fluent extensions on `Matrix<double>` and the structured solves |
| `src/Tensile.Kernels/KernelEntry.cs` | The single seam where spans are pinned and become pointers; restates every shape precondition |
| `src/Tensile.Kernels/Operand.cs` | `Operand` / `Target`: span + shape, length-checked on construction |
| `src/Tensile.Kernels/Alignment.cs` | Cache-line offset for pinned arrays; the one address read on the allocation path |
| `src/Tensile.Kernels/*.cs` | As before: kernels, packing, Gemm/ParallelGemm/GemmDispatch, ColumnOps, Pivoting, PanelProduct, Triangular, Lu, Norms, Reference |
| `src/Tensile.Interop.Blis/` | Native `bli_dgemm` binding + dispatch/ABI queries, its own package; `README.md` carries the `TENSILE_BLIS_LIBRARY` warning |
| `tests/Tensile.Fuzz/` | SharpFuzz harness: an input is a script of operations over hostile integers; the property is I5. Nightly under afl++; `--self-check` replays the seed corpus per PR |
| `tests/Tensile.Tests/` | xunit.v3, 965 tests; `Invariants/` is the secure-by-design spec, all green; kernel-generic contracts run per kernel via `IKernelCase` markers |
| `bench/Tensile.Benchmarks/` | BenchmarkDotNet: GEMM vs BLIS (one class, interleaved — see finding 12), serial vs threaded, kernel ceiling, LU block-size sweep, API overhead (what the security migration cost), thread scaling, serial/threaded crossover, serial cache-blocking sweep with both driver and dispatch arms |
| `tools/Tensile.Diagnostics/` | `tensile-diag`: ISA, BLIS dispatch, LU phase breakdown, estimator accuracy, expm accuracy against a Taylor oracle; and the codegen gate's process |
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
BLIS's 93%). **Two hypotheses for it have now been measured and both are
dead.** Cache blocking: the sweep below finds all four MC/KC combinations
within 1% of each other at n=128. Per-call overhead in the managed API
layers: the API-overhead measurement finds the shipped path no slower than
raw pointers at n=128 (in fact 7% faster, within noise).

Both of those tested things *above* the GEMM driver. What remains untested is
the driver itself at small n — packing setup and loop overhead that BLIS may
amortise better, or a small-matrix special path on its side. That is where to
look next, and nothing measured so far bears on it.

Note this table was taken with the old MC=288 and the serial default is now
MC=144, so it was re-run. **Both attempts are below, and between them they have
reopened the MC question rather than settling it.**

### Attempt 1, 2026-09-21: unpinned, and therefore void

The run was not pinned, so the environment and the parameter changed together.
The tell is inside the run itself: its threaded rows read 0.64 / 0.38 / 0.36 /
0.34 of serial at n=256 and up — a 2.6-3x speedup — and `taskset -c 0` makes
`ProcessorCount` 1, which makes that arithmetically impossible. C# read 38.7 /
44.3 / 49.0 / 46.4 / 45.9 and BLIS 48.3 / 46.7 / 46.5 / 43.9 / 44.1, both 13-23%
under the pinned table, at 4-11% StdDev against the pinned run's 1.2-2.0%. The
same run's *threaded* row at n=2048 read 137.9 GFLOP/s, exactly where every
other unpinned threaded measurement of this machine sits, so the machine was
not slow — the serial rows were. See finding 6.

### Attempt 2, 2026-09-21: pinned, and it disagrees with the block-size sweep

Pinned properly this time — the threaded rows read 0.99-1.05 of serial, which
is what `ProcessorCount == 1` looks like, and is the same tell read the other
way. BenchmarkDotNet's `DefaultJob` rather than the 31-iteration job, so fewer
samples than the methodology asks for; StdDev 1.9-6.5% (C#) and 2.2-5.1%
(BLIS). GFLOP/s from means.

| n | C# MC=144 | BLIS | ratio | ratio on 09-19 (MC=288) | change |
| --- | --- | --- | --- | --- | --- |
| 128 | 43.31 | 47.93 | 0.904 | 0.849 | **+0.055** |
| 256 | 46.54 | 45.90 | 1.014 | 0.972 | **+0.042** |
| 512 | 44.57 | 47.89 | 0.931 | 0.976 | -0.045 |
| 1024 | 44.73 | 49.59 | 0.902 | 1.025 | **-0.123** |
| 2048 | 45.49 | 50.94 | 0.893 | 0.997 | **-0.104** |

**Read the ratio column and nothing else: the absolute figures moved on both
sides again.** BLIS is an unchanged binary on an unchanged machine and it came
in 11-18% below its own 09-19 numbers, so this sitting was ~15% slower
throughout. That is the third recording of sitting-level drift larger than any
effect being chased, and it is now the single most expensive fact about
measuring on this machine.

**Within the sitting, the managed side gained 4-6 points of ratio at n<=256
and lost 10-12 at n>=1024.** That is the shape a smaller MC predicts, and it
has a mechanism. The packed A block is MC*KC*8 bytes: 442 KiB at MC=144 and
884 KiB at MC=288, against a Golden Cove P-core's 1.25 MiB L2. Pinned, one
thread owns all of it, so MC=288 fills it and MC=144 leaves half of it idle
while doubling the number of ic iterations — and each of those streams the
whole packed B panel again. MC=144 was derived for a Gracemont E-core's quarter
share of its cluster's 2 MiB L2, which is the right target for the threaded
path and quite possibly the wrong one for a pinned P-core.

**That explanation is now dead, and so is the alternative.** The block-size
sweep had A/B'd exactly these two MC values, pinned, in both directions, and
put MC=144 ahead by 9.5% at n=2048 — the opposite sign. The two runs differ in
which path they drive, the sweep the raw driver and this one the dispatch, and
those two had a long-standing unexplained 8-18% disagreement, so the dispatch
looked like the culprit. `BlockSizeBenchmarks` was extended to measure both
arms at each MC in one class, and it says otherwise:

- **the dispatch costs nothing** — six driver/dispatch pairs at ratios 1.003,
  0.966, 1.021, 0.985, 0.986, 0.991, sign flipping, every one inside its own
  row's StdDev;
- **MC is a wash at n=2048** in the same sitting — +1.3% for MC=144 on the
  driver and -1.7% on the dispatch, against 5-6.5% StdDev.

So neither the path nor the block size accounts for 10-12 points of ratio. What
remains is that `GemmBenchmarks` and `BlisGemmBenchmarks` were *separate
classes*, which BenchmarkDotNet runs as separate processes minutes apart, and
this machine moves ~20% between processes. **The ratio in the table above is
a cross-process ratio and is not trustworthy to better than about ten points.**
That is finding 12, and the fix is structural: the Tensile and BLIS rows now
live in one class (`GemmVsBlisBenchmarks`), interleaved by BDN over the same
operands, so the next parity figure is a within-process ratio.

The qualitative shape has replicated across three campaigns and survives all
of this — behind at n=128, level from n=256 up — but the decimal places in the
headline table were never as solid as they read, and the table needs re-taking
with the merged class before any of them is quoted again.

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

MC=144 against the old placeholder MC=288, best KC for each. **This table is
superseded by the re-run below — the n=2048 figure did not reproduce — and is
kept because what it got wrong is the lesson:**

| n | ascending | descending | verdict at the time |
| --- | --- | --- | --- |
| 128 | MC144 +13.8% | MC144 -0.7% | sign flips: thermal |
| 512 | MC144 +12.3% | MC144 -1.5% | sign flips: thermal |
| 2048 | MC144 +9.6% | MC144 +9.5% | ~~survives: real~~ **did not reproduce** |

The ascending-only run would have reported "MC=144 is 10-14% better at every
size". Two of those three numbers were the machine warming up, which is what
the reversal was for and what it caught. The third agreed to a tenth of a
percent in both directions and was recorded as settled; re-run in a fresh
sitting it measures roughly zero. Two directions run back to back rule out
the thermal gradient and nothing else — see finding 7.

MC=144 is the serial default, matching what `ParallelGemm` already derived
from cache geometry, and the re-run below supports keeping it on a different
and smaller effect. KC=256 and KC=384 came out within 1% of each other
everywhere in both directions, so KC stays at 384 and is no longer swept.

### Re-run 2026-09-21, both arms in one class, both directions

`BlockSizeBenchmarks` had an unexplained quarrel with `GemmBenchmarks`: the
same MC=288/KC=384 configuration ran 8-18% slower there, in every attempt,
including the clean high-priority one. Tiering was ruled out. It was written
off as thermal, and then the MC question came to rest on it, because the sweep
drives `Gemm.Multiply` with an explicit scratch while `GemmBenchmarks` goes
through `GemmDispatch`. So the class was given a dispatch arm and run both
ways. KC is fixed at 384, since 256 was within 1% of it everywhere. Pinned;
ascending on BDN's default job, descending at 31 iterations.

**The dispatch costs nothing, in both directions.** Twelve driver/dispatch
pairs, dispatch/driver from 0.966 to 1.021 — sign flipping, every one inside
its own row's StdDev, and the descending run's six all land in 0.977-1.009.
The 8-18% gap was drift between two processes, not a cost in the path every
caller takes, which is what `ApiOverheadBenchmarks` had already implied and
what the arithmetic says (O(1) checks against O(n^3) of work). Corroborating
it from the other side, this class's dispatch row at MC=144/n=2048 reads 44.48
against `GemmBenchmarks`'s 45.49 in the pinned sitting — two classes, two
sittings, 2.2% apart, where 09-19 had them 8-18% apart. See finding 12.

**MC=144 wins, but by a few percent and not nine.** Positive means MC=144
faster; means, with medians in the 09-19-style summary below.

| n | arm | ascending | descending | mean of the two |
| --- | --- | --- | --- | --- |
| 128 | driver | +1.2% | +0.5% | +0.9% |
| 128 | dispatch | -0.6% | +0.1% | -0.2% |
| 512 | driver | +4.9% | +3.0% | **+3.9%** |
| 512 | dispatch | +7.0% | +1.8% | **+4.4%** |
| 2048 | driver | +1.3% | +2.8% | +2.1% |
| 2048 | dispatch | -1.7% | +2.0% | +0.2% |

**The descending run is the one that carries this**, because BDN sorts by
parameter value and the reversal therefore runs MC=288 first and coolest:
MC=144 wins all six descending cells while running last and hottest, which is
the strong form of the test (the same argument that makes LU's nb=32 result
solid). Eleven of the twelve cells are positive.

But the magnitude is small and size-dependent. **n=512 is the only size with a
real effect**, ~4% averaged and positive on both arms in both directions.
n=128 is nothing. **n=2048 is a wash** — +2.1% and +0.2% averaged, against
5-6.5% StdDev, which is the size where this machine is noisiest.

**So the 9.5% at n=2048 recorded above does not reproduce.** That figure came
from a pair of runs whose two directions agreed to a tenth of a percent, and
it is now measured at roughly zero by a fresh pair that also agree with each
other. Bidirectional agreement inside one sitting rules out the thermal
gradient; it does not make an effect reproducible across sittings. See
finding 7.

**MC=144 stays, now on evidence rather than on a number that evaporated**: no
worse than MC=288 at any size in either direction, ~4% better at n=512, and it
is also what `ParallelGemm` derives from cache geometry, so the two paths
agree. What is still unknown is whether some third value beats both — MC has
only ever been measured at two values and three sizes, and NC has never been
varied at all.

Absolute figures, for the record and not for comparison: 40.2-41.2 GFLOP/s at
n=128, 43.4-45.1 at n=512, 44.7-46.0 at n=2048. MC=288 at n=2048 reads 44.7
here against `GemmBenchmarks`'s 57.22 on 09-19 — identical configuration, 22%
down. The sitting moved again, by far more than anything being measured
inside it. Note also that going from the default job to 31 iterations barely
touched dispersion at n=2048 (6.2-6.5% to 5.1-6.6%), so the sample count is
not what limits that size; the machine is.

## What the secure-by-design migration cost: nothing measurable

The section 9 guardrail of `docs/security-design.md`, and the last number the
security work was resting on. Three rows over the same GEMM driver, differing
only in what sits above it: native aligned memory straight into the dispatch;
a `Matrix<double>` on the pinned object heap, still straight into the
dispatch; and the shipped path — `Workspace.Multiply` over bound views, which
adds `Bind`, the `Operand` length checks, the workspace lock and the pinning
seam. Unpinned, root, 31 iterations, GFLOP/s from medians.

| n | native pointers | POH storage, direct | public API | StdDev range |
| --- | --- | --- | --- | --- |
| 128 | 43.31 | 45.39 (+4.8%) | 46.41 (**+7.2%**) | 8.3-9.5% |
| 512 | 140.44 | 129.55 (-7.8%) | 131.06 (**-6.7%**) | 2.0-6.2% |
| 2048 | 136.81 | 137.75 (+0.7%) | 137.47 (**+0.5%**) | 1.8-3.1% |

**The sign flips across sizes — +7.2%, -6.7%, +0.5% — which is noise around
zero, not a cost.** At n=512, the one size where the API looks slower, the
7% gap is 1.1 standard deviations of the *native* row, which has the worst
dispersion in that group (6.2%). At n=2048, where dispersion is tightest, the
three paths agree to half a percent.

Two things make this stronger than a bare tie. First, the execution order
within each size is native, then POH, then API, so the shipped path runs last
and hottest; the thermal gradient works against the conclusion rather than
producing it. Second, the arithmetic was never in doubt — every check the
migration added is O(1) per call against O(n^3) of work — so the measurement
was confirming an expected zero rather than hunting for a small effect. A
ratio that had *failed* to converge would have meant per-element work, which
would be a defect.

So: bounds-checked views, validated shapes, pinned-object-heap storage, a
single pinning seam and a workspace lock, for no measurable throughput. That
is the whole case for the secure-by-design rewrite, measured rather than
asserted.

One caveat of the usual kind: absolute figures here (136.8 GFLOP/s threaded
at n=2048) sit ~7% below the thread sweep's 147.0 at the same thread count,
which is the same cross-benchmark-class drift recorded elsewhere. The
within-run comparison is what carries the result.

## Serial vs threaded: the crossover is 2^24, and it has no shape term

`ParallelThreshold` decides serial against threaded for every product,
including every trailing update in LU. It was 4e6, derived from a fork/join
cost argument and never checked. Measured by running both paths explicitly
across shapes bracketing the crossover, in both directions, unpinned, root,
31 iterations, ratios from medians.

| work (m·n·k) | shape | ascending | descending | mean | winner |
| --- | --- | --- | --- | --- | --- |
| 262,144 | 64 | 1.056 | 1.267 | 1.162 | serial 16% |
| 884,736 | 96 | 1.144 | 1.160 | 1.152 | serial 15% |
| 2,097,152 | 128 | 1.048 | 1.085 | 1.066 | serial 7% |
| 4,096,000 | 160 | 1.182 | 1.091 | 1.136 | serial 14% |
| 4,194,304 | 256x64 | 1.258 | 1.089 | 1.173 | serial 17% |
| 7,077,888 | 192 | 1.073 | 1.207 | 1.140 | serial 14% |
| **16,777,216** | 256 | 0.702 | 0.646 | 0.674 | **threaded 33%** |
| **16,777,216** | 512x64 | 0.557 | 0.581 | 0.569 | **threaded 43%** |
| 56,623,104 | 384 | 0.441 | 0.419 | 0.430 | threaded 57% |
| 67,108,864 | 1024x64 | 0.321 | 0.328 | 0.324 | threaded 68% |
| 268,435,456 | 2048x64 | 0.358 | 0.318 | 0.338 | threaded 66% |

**All eleven shapes picked the same winner in both directions — no
disagreement anywhere.** Sorted by work the transition is perfectly clean:
serial ahead at every work below 2^24, threaded ahead at every work at or
above it, with no overlap. After two sweeps in this campaign where one
direction misled, this one replicated exactly.

`DefaultParallelThreshold` is now **16,777,216**. The crossover lies in
(7,077,888, 16,777,216]; 2^24 is the conservative end, being the smallest
work threading was actually observed to win. The old 4e6 sat below the
crossover and threaded three measured shapes that lose 14-17% by it.

**The threshold needs no shape term, which the sweep was built to find out.**
Square operands and the `m x m x 64` panels LU's trailing update produces were
expected to disagree — a panel carries far more memory traffic per flop — so
both families were measured. They agree exactly: 256^3 and 512^2*64 are both
2^24, and both are the first threaded win in their family. Work alone
predicts the path.

For LU this moves where the trailing update stops threading. At nb=64 the old
threshold threaded until the trailing block was 250x250; the new one stops at
512x512, leaving the tail serial where serial is measured to be faster.

## Threading: power-limited, and the baseline is a trap

Re-taken on the 12700H, 2026-09-19: unpinned (finding 6), root so BDN could
raise priority, 31 iterations, `ParallelGemm` driven directly so the dispatch
threshold cannot divert a row. **Run in both directions** and averaged, since
thread count and execution order are the same variable in an ascending sweep.
GFLOP/s from means (the descending report carried no median column).

| threads | n=2048 asc | n=2048 desc | n=2048 mean | n=512 mean |
| --- | --- | --- | --- | --- |
| 1 | 45.34 | 45.94 | 45.64 | 52.46 |
| 2 | 86.42 | 87.70 | 87.06 | 81.57 |
| 4 | 133.25 | 130.20 | 131.72 | 115.15 |
| 6 | 170.43 | 162.52 | **166.47** | 128.93 |
| 8 | 151.17 | 146.46 | 148.81 | **144.64** |
| 12 | 145.88 | 148.32 | 147.10 | 128.80 |
| 16 | 140.55 | 143.96 | 142.25 | 124.20 |
| 20 | 142.20 | 151.81 | 147.00 | 126.68 |

**Peak at 6 threads for n=2048 and 8 for n=512, picked independently by both
directions.** The location is solid; a third campaign (the original one) also
peaked at 6. ~145-165 GFLOP/s at 6-8 threads is this machine.

**The decline past the peak is real, but a one-directional sweep overstates
it by about half.** Off-peak at 20 threads reads 17-19% ascending, 6-7%
descending, and **12% averaged**. The crossover signature is textbook: at 6
threads the ascending run reads higher (it ran early and cool), at 20 threads
the descending run reads higher (same reason), and averaging cancels it. The
per-configuration spread between directions was 1.3-12.3%, which is a
useful measure of the thermal gradient across one sweep on this part.

**The 1-thread row is not a valid baseline and speedups quoted against it are
inflated.** Unpinned, a `Parallel.For` with `MaxDegreeOfParallelism = 1` is
scheduled wherever the OS likes, including onto a Gracemont E-core. It reads
45.34 ascending and 45.94 descending — reproducible, and both far below the
57.22 measured pinned on a P-core, so it understates the baseline by ~25% and
overstates every speedup by the same factor. Against the pinned figure the
real peak speedup is **2.91x on 6 threads, 49% per-thread efficiency**, not
the 3.6x the in-run baseline suggests.

This is the sharp edge of finding 6: you must not pin a scaling sweep, and
the 1-thread row of an unpinned sweep is therefore worthless as a
denominator. Take the baseline from a separate pinned run.

## LU

Re-taken on the 12700H, 2026-09-19: threaded (the trailing update goes through
`GemmDispatch`, so sizes above `ParallelThreshold` use every core), unpinned,
31 iterations, GFLOP/s from medians at (2/3)n^3 - n^2/2 - n/6 flops.

Run in both directions and averaged (GFLOP/s from means, which is what the
descending report carried):

| n | nb=32 | nb=64 | nb=128 | best, both ways? |
| --- | --- | --- | --- | --- |
| 256 | **20.37** | 18.35 | 13.89 | nb=32, **+11.0%** |
| 512 | **28.55** | 25.37 | 21.39 | nb=32, **+12.6%** |
| 1024 | 40.43 | **42.29** | 34.52 | nb=64, **+4.6%** |
| 2048 | 65.25 | **65.67** | 52.21 | tie, 0.6% apart |

The old container table (18.9 / 24.5 / 26.7 at 512 / 1024 / 2048, single
virtualised AVX2 core) is superseded and not comparable — different machine,
different thread count.

**Three of the four sizes pick the same winner in both directions, and the
small-n results are the strongest of them.** BenchmarkDotNet runs nb=32 first
ascending and last descending, so in the descending sweep nb=32 wins at n=256
and n=512 while running last and hottest — it beats the thermal gradient
rather than riding it. nb=64's 4.6% at n=1024 is likewise confirmed both
ways. At n=2048 the directions disagree over a 0.6% gap, which is a tie.

nb=128 is worst at every size in both directions, by 19-33%, and that has a
mechanism as well as a measurement: the panel factorization is O(m*nb^2)
level-2 work, so a wider panel moves more of the total into the slow
unblocked path.

**The default is now size-dependent**: `Lu.DefaultBlockSizeFor` returns 32
below order 1024 and 64 at or above it. The previous flat nb=64 was costing
11-13% on every factorization below n=1024. The crossover sits somewhere in
(512, 1024]; 1024 is the conservative end, since that is where nb=64 was
actually measured to win.

**Against threaded GEMM at the same thread count, LU is at 45%** — and the
arithmetic explains the whole gap without implicating the factorization.

| n | LU (best nb) | threaded GEMM, 20 threads | ratio |
| --- | --- | --- | --- |
| 512 | 28.55 | 126.68 | 23% |
| 2048 | 65.67 | 147.00 | **45%** |

(GEMM figures averaged over both sweep directions.) LAPACK's norm is 70-80%,
so 45% looks alarming. It is not a defect, it is Amdahl, and the numbers
close:

- Threaded GEMM is 2.57x single-core GEMM (147.00 / 57.22).
- Apply that to the GEMM share alone of the phase breakdown below — 65% GEMM,
  35% panel + swaps + TRSM, all three of which are serial — and the
  factorization should speed up by 100 / (65/2.57 + 35) = **1.66x**.
- Working backwards from the observed 65.52 threaded, serial LU would be
  **39.5 GFLOP/s, or 69.1% of serial GEMM** — squarely inside LAPACK's band.

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
Re-taking it is the highest-value LU measurement outstanding, because it is the
input to the Amdahl argument above.

**The instrument for re-taking it now exists**: `Lu.Factor` takes an optional
`LuPhaseTimings`, and `tensile-diag` prints the split at n=512/1024/2048 in its
default (non-`--quiet`) run. Two things to know before quoting a number from it.

First, *the environment chooses which measurement you get*: pinned
(`taskset -c 0`) gives the serial split, unpinned the threaded one, and they
answer different questions. The Amdahl argument wants the threaded one at the
thread count LU actually runs at; the serial one is what the 65/14/10/11 row
above is, and re-taking that too would let the two be compared directly.

Second, the report carries its own control. The collector is opt-in, so the
tool factors each size both ways — alternating, comparing medians, because the
quantity at stake is a fraction of a percent and running one block then the
other would charge the whole thermal gradient to the instrument. The
`instrument` column is that difference. On the 4-core development container it
reads -2.2% to +7.9% at n=512 and within +-2% above, i.e. noise straddling
zero, while the shares themselves reproduce to a few tenths of a percent
between runs. If that column ever reads consistently positive, the split is
measuring the instrument and not the factorization.

An `unattr` column reports loop time the four phases do not claim (the pivot
fix-up, the diagonal scan, loop overhead). It is deliberately printed rather
than normalised away: it read 0.5-0.6% at n=512 and 0.0% above on the
development container, and a split that silently summed to 100% could hide a
phase going unaccounted.

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

   The corollary bites on a hybrid part: **any single-threaded row of an
   unpinned run is worthless as a baseline** — and this is not specific to
   `Parallel.For`, which was how it was first found. The serial GEMM path,
   which forks nothing at all, measured 12.9-21.9% below its pinned figure in
   the unpinned 2026-09-21 re-run, with dispersion going from 1.2-2.0% StdDev
   to 4-11%. In that same run the *threaded* row at n=2048 was unaffected
   (137.9 GFLOP/s, in line with every other unpinned measurement), which is
   the shape to recognise: a run where the parallel rows look normal and the
   serial rows look 20% slow is an unpinned run, not a regression.

   A `Parallel.For` limited to one worker
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

   Reversal does not always overturn a result; the second sweep run both ways
   shows the other outcome. In the thread-count sweep the peak location (6
   threads at n=2048, 8 at n=512) came out identical in both directions, so it
   is solid — but the *magnitude* of the post-peak decline read 17-19%
   ascending, 6-7% descending, and 12% averaged. The one-directional figure
   was roughly twice the truth. So: reversal can flip a sign (block sizes) or
   merely halve a magnitude (threads), and averaging the two directions is the
   cheapest way to get a number worth quoting. Per-configuration spread
   between directions was 1.3-12.3% here, which is the thermal gradient's
   size on this part.

   **And the limit of the test, which cost a conclusion to learn: agreeing in
   both directions is not the same as reproducing.** The block-size sweep's
   MC=144 win at n=2048 read +9.6% ascending and +9.5% descending — a tenth of
   a percent apart, which is as convincing as a bidirectional result ever
   looks, and it was recorded as settled. Re-run months later it measures
   +2.8% and -1.7%, essentially zero, with the two new directions agreeing
   with *each other*. Two directions run back to back share everything except
   the thermal gradient, so their agreement rules out the gradient and nothing
   else. A result is reproducible when a *different sitting* finds it, and on
   this machine a different sitting is worth up to 22%.

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
    `LuTest.cs`, `Blas1.cs` (now `ColumnOps.cs`/`Pivoting.cs`) and `Triangular.cs`
    all arrived in one commit that
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

12. **A ratio computed across two benchmark classes is a ratio across two
    processes.** BenchmarkDotNet runs each class in its own generated process,
    minutes apart, and on this 12700H the same binary configuration has read
    up to 22% differently between two sittings and 8-18% differently between
    two processes in one sitting. Two conclusions were drawn from cross-class
    ratios and both were wrong: an 8-18% cost in `GemmDispatch` that vanished
    to 0.966-1.021 when both arms ran in one class, and a 10-12 point drop in
    the parity ratio against BLIS blamed on MC=144, which a within-class A/B
    then measured as +1.3%/-1.7% at that size. Neither effect existed.

    The rule: **a ratio is only as good as the interleaving behind it.** If two
    rows are to be divided, BDN must have run them back to back over the same
    operands, which means putting them in the same class. That is why the
    Tensile and BLIS rows now live together in `GemmVsBlisBenchmarks` rather
    than in two classes compared by hand, and why `BlockSizeBenchmarks` carries
    its dispatch arm.

    The tell that a cross-class comparison has gone wrong is an unchanged
    control moving: BLIS is an unchanged binary, and any run where it reads
    10-20% off its previous figure is a run whose absolute numbers are not
    comparable with anything from another sitting. Check the control before
    reading the experiment.

13. **`expm` has two different constants both called theta_13, and picking the
    wrong one is silent.** Table 3.1 of Al-Mohy & Higham (2009) gives
    `theta_13 = 5.371920351148152`, the norm below which the degree-13
    approximant meets the backward error bound. Algorithm 3.1 then chooses the
    scaling parameter from `s = ceil(log2(eta_5 / theta_13))` — with **4.25**,
    a deliberately smaller number. Using the table value there underscales by
    one step on some inputs, which does not fail any structural test and does
    not throw; it just quietly costs digits on exactly the nonnormal matrices
    the 2009 algorithm exists to handle.

    This was caught by checking the reference implementation rather than by
    reasoning, and it is the reason the constant is named `ScalingThreshold` in
    the source instead of `Theta13`. The general lesson for porting numerical
    papers: a constant that appears twice with the same name in the literature
    is a place to verify, not to infer, and the cost of getting it wrong is
    measured in digits rather than in exceptions.

    The defence that actually generalises is the oracle. A wrong threshold, a
    mistyped Padé coefficient, or a sign error in the `ell` correction all
    produce a plausible-looking matrix. What distinguishes them is an
    independent implementation, and for the exponential a 40-term Taylor series
    at `||A||/2^s <= 1/32` is one: obviously correct, far too slow to ship, and
    sharing nothing with the Padé path but GEMM.

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

- **Kernel primitives are named for their role, not for a BLAS level.** BLAS
  classifies by operand arity, which was a way to fit names into a flat Fortran
  symbol table; the axis that actually determines the code here is **packed vs
  streamed**. `Gemm` packs into micro-panels because O(n^3) of work amortises
  it. `ColumnOps`, `Pivoting`, `PanelProduct`, `Triangular` and `Norms` stream
  their operands in place because nothing they do would ever pay for packing.
  So `Blas1` became `ColumnOps` (the shared vectorised column updates) plus
  `Pivoting` (a *search*, whose contract is the index it returns, not a
  residual), and `Blas2` became `PanelProduct.Apply`/`ApplyTranspose`, named
  for `ILinearOperator.Apply` — its only consumer, through
  `KernelEntry.MultiplyPanel`. The old name was wrong on BLAS's own terms
  anyway: `Y := A*X` for an n x t panel is a matrix-matrix product that happens
  to be computed a column at a time, not a level-2 operation.
  `Blas1.MaxAbsStrided` was deleted in the same pass — it had no caller
  anywhere, which is finding 10 exactly.

- **The operator interfaces are split by capability.** `ILinearOperator`
  requires `Apply` only; `ITransposableOperator` adds `ApplyTranspose`, and
  that is what `NormEstimate` takes, because Higham and Tisseur's estimator
  alternates products with A and A^T. Requiring both on one interface is the
  same mistake as a `trans` flag on every BLAS signature: it makes every
  implementer carry what one algorithm needs. A matrix-free FEM or MTL
  operator — the actual target application — frequently applies A cheaply and
  cannot apply A^T at all, and `expmv` never needs the transpose. Split before
  `expmv` was written, while the interface was still cheap to change.

---

# Open items

- ~~**`Gemm.cs` uses placeholder MC=288, KC=384.**~~ *Closed, on the second
  attempt and with a much smaller number than the first.* The 09-19 sweep's
  headline — MC=144 ahead 9.5% at n=2048, agreeing in both directions — did
  not reproduce; the 09-21 re-run, both arms and both directions at 31
  iterations, measures that size as a wash. What survives is a consistent
  small win: eleven of twelve cells positive, all six positive in the
  descending run where MC=144 runs last and hottest, ~4% at n=512 and ~0-2%
  elsewhere. MC=144 stays. KC is settled at 384. **What is still open is
  narrower than before**: MC has been measured at exactly two values and three
  sizes, NC=4096 has never been varied at all, and a size-dependent MC has
  never been tried.
- ~~**Small-n threading.**~~ *Mostly closed.* The dispatch now keeps
  everything below 2^24 of work on the serial path, which is measured correct
  at every size tested from n=64 to n=192 (serial ahead 7-17%). n=128 in
  particular is 2.1e6 of work and firmly serial. What remains unexamined is
  the *other* small-n control, the worker-count clamp inside
  `ParallelGemm.Multiply` — `Math.Clamp((int)(work / 8_000_000), 1,
  scratch.MaxThreads)`. Its divisor of 8M is still derived rather than
  measured, and it now only takes effect above 2^24, where it caps a 2^24
  product at 2 threads. Whether that is the right cap is untested.
- ~~**`GemmDispatch.ParallelThreshold` (4e6 flops) is derived, not measured.**~~
  *Closed.* Measured in both directions with eleven shapes and zero
  disagreements; the crossover is 2^24 and the threshold is now set there. The
  two shape families turned out to agree exactly, so no shape term is needed.
  See "Serial vs threaded" above. What is still unmeasured is the band
  (7.08e6, 1.68e7] itself — no shape was tested inside it, so the true
  break-even could be anywhere in there, and 2^24 is the conservative choice
  rather than the optimal one.
- ~~**LU's block size is settled only above n=1024.**~~ *Closed.* Run both
  ways: nb=32 wins at n=256 and n=512 in both directions (and in the
  descending one it wins while running last and hottest), nb=64 wins at
  n=1024 both ways, and n=2048 is a tie. `Lu.DefaultBlockSizeFor` is now
  size-dependent — 32 below order 1024, 64 at or above — where the flat nb=64
  had been costing 11-13% on smaller factorizations. What is still unmeasured
  is the crossover's exact location, which lies somewhere in (512, 1024].
- ~~**The 8-18% gap between driving the GEMM driver and driving the dispatch is
  unexplained.**~~ *Closed: it was never there.* Measured with both arms in one
  class at each MC — six pairs, dispatch/driver 0.966 to 1.021, sign flipping,
  every one inside its own StdDev. It was drift between two processes. See
  finding 12, which is the general form and cost two conclusions before it was
  understood.
- ~~**The thread sweep's tail needs a descending run.**~~ *Closed.* Run both
  ways and averaged: peak at 6 threads (n=2048) and 8 (n=512), identical in
  both directions; decline past the peak is 12%, not the 17-19% the ascending
  run alone reported. What remains unmeasured is the E-core / P-core / SMT
  decomposition of the old table — this sweep varied only the worker count,
  not which cores it was allowed to use.
- **The LU phase breakdown is single-core and from the old container.** It is
  now load-bearing — the Amdahl argument that explains LU's 46% of threaded
  GEMM rests on the 65/14/10/11 split — so re-taking it threaded on this
  machine is the highest-value LU measurement outstanding. The instrumentation
  for it has landed (`LuPhaseTimings`, reported by `tensile-diag`); what is
  missing is a run on the 12700H, pinned and unpinned, which is two invocations
  and no rebuild. See "LU" above for how to read the report.
- **`normest1` has not been cross-validated against MATLAB's `normest1` or
  LAPACK's `dlacn2`.** It is verified by invariants instead — see finding 8 —
  which is strong evidence but not the same thing.
- **The estimator's `PanelProduct` applications are O(n^2 t) with no
  blocking.** Fine at the sizes that matter for `dgecon`, possibly not for
  `expm`'s inner loop.
- ~~**What the secure-by-design migration cost is unmeasured.**~~ *Closed.*
  Measured on the 12700H: nothing, at any size. +7.2%, -6.7%, +0.5% at
  n=128/512/2048 — noise around zero, with the shipped path executing last and
  hottest in every group. See "What the secure-by-design migration cost"
  above. The section 9 guardrail is satisfied.
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

1. ~~**`expm`**~~ *Done.* Al-Mohy & Higham (2009), the full degree
   3/5/7/9/13 ladder with the `ell` correction, `Expm` on `Matrix<double>`.
   The `||A^k||^(1/k)` estimates go through `DenseMatrixOperator` raised to a
   power, so no power of A is formed to measure it — which is what that
   parameter was built for. Verified three ways: the coefficient tables are
   re-derived from their closed forms, closed-form exponentials pin the
   analytic cases, and everything else is compared against an independent
   Taylor oracle. Accuracy runs 2e-16 to 4e-13, degrading with the squaring
   count and not with the degree, which is the shape backward stability
   predicts. **What is not done**: no benchmark (so the 15-25 products
   estimate is arithmetic, not measurement), no Schur-Parlett fallback for
   the badly nonnormal case, and the estimator's probe count is left at the
   default 2 rather than tuned for this use.
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
  `Packing`, `ColumnOps`, `Pivoting`, `PanelProduct` and `Triangular` are
  exactly where an off-by-one hides.
- **`tensile-diag`** covers what unit tests cannot: which kernels this host
  actually has, what BLIS dispatched to, estimator accuracy by ensemble, and
  where a blocked LU spends its time — numbers worth reporting rather than
  asserting. The LU split is there rather than in the benchmarks because
  BenchmarkDotNet measures a whole call and cannot see inside one; what the
  test suite pins about it is structural (the phases sum to no more than the
  loop they were charged out of, the step count matches the block size, and
  collecting the timings leaves the factorization bit-identical), never a
  duration.
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
- **Port a numerical paper against an oracle, not against a reading of it.**
  `expm`'s coefficient tables are re-derived from their closed forms in the
  suite, so a transcription slip fails rather than costing accuracy; the theta
  thresholds cannot be derived that way, so what guards them is comparison
  against an independent Taylor implementation. See finding 13.
- **A branch with no test is as unverified as a file with no caller.** `expm`
  picks one of five Padé degrees, and a suite whose matrices all landed on
  degree 13 would leave four branches unexecuted while passing. That is why
  `ExpmDiagnostics` exists and why `EveryPadeDegreeIsReached` asserts the set
  of degrees observed, not just that the answers were right. Finding 10,
  applied one level down.
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
  latency. **Pass `--iterationCount 31` explicitly** — nothing in the config
  sets it, so a run that does not say so on the command line gets BDN's
  `DefaultJob`, which does not resolve single-digit-percent effects on this
  machine. The report header says which you got.
- **Never divide two numbers from two benchmark classes.** Separate classes are
  separate processes run minutes apart, and this machine drifts 8-22% between
  them; two conclusions have already been destroyed that way. Rows that will be
  divided go in one class so BDN interleaves them. See finding 12.
- **Read the control first.** BLIS is an unchanged binary: if it reads 10-20%
  off its previous figure, nothing in that run is comparable with anything from
  another sitting, and only within-run ratios mean anything.
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
