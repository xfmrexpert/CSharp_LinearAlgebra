# Tensile

Dense linear algebra for .NET, real and complex, in managed code: blocked GEMM
with hand-written micro-kernels at parity with BLIS on one core, LU with partial
pivoting, 1-norm condition estimation, and the matrix exponential and its action.
No native dependencies and no NuGet packages in the library itself.

It started as one question — **does RyuJIT keep a full set of vector
accumulators in registers across a GEMM micro-kernel's k-loop, or does it
spill?** — and the answer was good enough to build on. The eventual target is
transient analysis of power transformer windings, which means dense
complex-valued frequency-domain matrices and their exponentials at orders up to
about a thousand.

```csharp
using System.Numerics;
using Tensile;

var a = Matrix.FromRows(new[,] { { 4.0, 1.0 }, { 1.0, 3.0 } });
var b = Matrix.FromColumnMajor<double>(2, 1, [1.0, 2.0]);

Matrix<double> x = a.Solve(b);
double rcond = a.FactorLu().ReciprocalCondition();
Matrix<double> e = a.Expm();                        // the matrix exponential

var z = Matrix.FromRows(new Complex[,] { { new(0, 1), 2.0 }, { -2.0, new(0, 1) } });
Matrix<Complex> y = z.Expmv(z.ReadOnlyView, t: 0.5); // exp(0.5 Z) Z, without forming exp
```

**[API guide](docs/api.md)** — matrices and views, structure-typed dispatch,
factorizations, norms and conditioning, workspaces and threading, matrix-free
operators, the exponentials, and complex matrices.

## What is in it

- **Matrices and views.** `Matrix<T>` owns pinned, 64-byte-aligned storage and
  is not disposable; `MatrixView<T>` is a bounds-checked `Span`-backed view.
  Column-major, unit row stride.
- **GEMM.** BLIS-style blocking with AVX-512, AVX2/FMA and scalar
  micro-kernels selected per host, single- and multi-threaded.
- **Complex products** by the 4M method over the real GEMM — not 3M, which
  loses a small imaginary part (the damping in a line model) to cancellation.
- **LU with partial pivoting**, real and complex, behind one
  `LuDecomposition<T>`; solves with A and its adjoint; determinant.
- **Condition estimation**: Higham–Tisseur block 1-norm estimation
  (`normest1`), and the `dgecon`/`zgecon` equivalent built on it.
- **Structure in the type system**: triangular structures dispatch a solve to
  substitution at compile time.
- **The matrix exponential** (Al-Mohy & Higham 2009) and **its action**
  exp(tA)B (Al-Mohy & Higham 2011), real and complex, dense and matrix-free
  over `ILinearOperator<T>`.

Not here yet: Cholesky, QR, SVD, eigenvalues. See the end of the API guide.

## Secure by design

The public assembly compiles with unsafe code disallowed and integer overflow
checking on; every pointer lives in an internal kernel assembly reached through
one pinning seam. Every allocation sized by a request passes a process-wide
ceiling (`TensileLimits.MaxElements`). The public surface is fuzzed nightly. The
design, and the measurement that it costs nothing in throughput, are in
[`docs/security-design.md`](docs/security-design.md).

## Layout

| Path | Contents |
|---|---|
| `src/Tensile` | The public assembly: matrices, views, structures, factorizations, norms, operators, exponentials. No unsafe code. |
| `src/Tensile.Kernels` | The kernel assembly: micro-kernels, packing, GEMM, LU, triangular solves. All `internal`; all the unsafe code. |
| `src/Tensile.Interop.Blis` | The optional native BLIS binding, a separate package, for benchmarks. The core never references it. |
| `tests/Tensile.Tests` | xunit.v3 suite. Contracts generic over the micro-kernel run once per kernel the host supports. |
| `tests/Tensile.Fuzz` | SharpFuzz harness over the public surface, run nightly under afl++. |
| `bench/Tensile.Benchmarks` | BenchmarkDotNet: GEMM against BLIS, kernel ceiling, threading, LU, API overhead, complex products and exponentials. |
| `bench/run-measurements.sh` | Runs the measurements the project is waiting on, pinned or unpinned as each requires, into a dated directory. |
| `tools/Tensile.Diagnostics` | `tensile-diag`: host ISA, BLIS dispatch, kernel correctness checks, LU phase split, estimator and exponential accuracy. |
| `disasm.sh` | Dumps micro-kernel codegen and fails on accumulator spills. CI runs it. |

## Building and testing

The project targets `net10.0` and needs the .NET 10 SDK.

```
dotnet test --solution Tensile.slnx -c Release              # the unit suite
dotnet run -c Release --project tools/Tensile.Diagnostics   # what this host supports, and the reports
./disasm.sh                                                 # the codegen gate
```

Note `--solution`: under Microsoft.Testing.Platform the positional form
`dotnet test Tensile.slnx` is rejected.

## Performance

On an Intel i7-12700H P-core (AVX2; AVX-512 is fused off on that part), pinned,
against BLIS 0.9.0 dispatching to its `haswell` assembly kernels — the same
6x8 AVX2 geometry, so the same algorithm compiled two ways:

| n | 128 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|
| Tensile / BLIS | 85% | 97% | 98% | 103% | 99.7% |

The managed GEMM reaches ~91% of its own L1-resident kernel ceiling, as BLIS
does, and the micro-kernel issues at ~98% of the core's FMA peak. The shape —
behind at n=128, level from 256 up — has replicated across three campaigns, but
the decimal places came from two benchmark processes compared by hand, which on
that machine can drift 10–20% apart; the table is being re-taken with both
sides in one interleaved benchmark class (`GemmVsBlisBenchmarks`). The full
record, including what went wrong in measuring it, is in `CLAUDE.md`.

Benchmarking on a laptop part lies easily. The rules the harness is built
around:

- **Pin single-thread comparisons** (`taskset -c 0`), and **do not pin scaling
  runs** — `Environment.ProcessorCount` respects the affinity mask.
- **Pass `--iterationCount 31`**; nothing in the configuration sets it.
- **Only divide numbers from the same benchmark class**, which BenchmarkDotNet
  runs interleaved in one process.
- **Run parameter sweeps in both directions** (`TENSILE_BENCH_REVERSE=1`); a
  one-directional sweep confounds its parameter with the machine warming up.

`bench/run-measurements.sh` applies them.

### Native BLIS comparison

The binding is its own package, `Tensile.Interop.Blis`, referenced by the
benchmarks and `tensile-diag` and never by the library. Read
[`src/Tensile.Interop.Blis/README.md`](src/Tensile.Interop.Blis/README.md)
before pointing `TENSILE_BLIS_LIBRARY` at anything on a machine you do not
control.

```bash
sudo apt install libblis-dev                                  # Debian/Ubuntu/Pop!_OS
TENSILE_BLIS_LIBRARY=/absolute/path/to/libblis.so \
  dotnet run -c Release --project tools/Tensile.Diagnostics   # a custom build instead
```

`tensile-diag` reports the architecture BLIS dispatched to and its micro-kernel
classification. A `generic` architecture or an unoptimized kernel means the
comparison is against a fallback: rebuild BLIS with `./configure auto` and
check again before reading any ratio. Without BLIS, the managed benchmarks
still run and the comparison class is skipped.

## Finding 1: no spills — the generated inner loop is what you would hand-write

The AVX-512 k-loop, from `DOTNET_JitDisasm`:

```
G_M000_IG03:
       vmovups  zmm17, zmmword ptr [rsi]
       vmovups  zmm18, zmmword ptr [rsi+0x40]
       add      rsi, 128
       vbroadcastsd zmm19, qword ptr [rdx]
       vfmadd231pd zmm1, zmm19, zmm17
       vfmadd231pd zmm2, zmm19, zmm18
       vbroadcastsd zmm19, qword ptr [rdx+0x08]
       ...
       add      rdx, 64
       inc      eax
       cmp      eax, edi
       jl       G_M000_IG03
```

Per k iteration: 16 `vfmadd231pd`, 8 `vbroadcastsd` with folded memory
operands, 2 `vmovups`, 4 scalar ops of loop overhead, and no stack traffic —
all 16 accumulators stay in `zmm1`–`zmm16`. **RyuJIT is not the bottleneck
for a BLIS micro-kernel.**

## Finding 2: the register allocator is shape-sensitive, and fails quietly

An earlier probe with 16 independent FMA chains in a flat loop and a
loop-invariant constant — same register count, same instruction set — spilled
every accumulator:

```
       vmovups  zmm2, zmmword ptr [rbp-0xB0]
       vfmadd213pd zmm2, zmm0, zmmword ptr [reloc @RWD00]
       vmovups  zmmword ptr [rbp-0xB0], zmm2
```

A load/FMA/store round trip per accumulator per iteration, about 5x off peak,
with no warning. **Register residency cannot be assumed from the source**, so
the disassembly check is in CI. The spills were `rbp`-relative, not
`rsp`-relative; grep for both.

## Finding 3: tiered compilation will lie to you

Large GEMMs make too few calls into the kernel to tier up inside a measurement
window: without `AggressiveOptimization` the first run read 14–16 GFLOP/s at
n=128–1024 and 73 at n=2048. Every hot path carries the attribute; when timing
by hand, cross-check with `DOTNET_TieredCompilation=0`.

## Reading the disassembly yourself

```bash
./disasm.sh kernel.asm
```

It dumps the kernels, reports FMA and spill counts per kernel, and exits
non-zero if any accumulator reached the stack. Two traps it exists to avoid:
do not dump through `dotnet run`, whose SDK driver and application both honour
`DOTNET_JitStdOutFile` and overwrite each other's output, so kernels go missing
without an error; and scope the check to the kernel listings, since plenty of
framework methods are also called `Execute`. On .NET 10.0.112 the totals,
write-back included, are 32 `vfmadd` for the AVX-512 16x8 kernel and 24 for
the AVX2 8x6, with zero spills.

## Project notes

`CLAUDE.md` is the working record: every measurement with its conditions, the
findings that cost something to learn, the design decisions and why, and the
open items.

## Licence

See [LICENSE](LICENSE).
