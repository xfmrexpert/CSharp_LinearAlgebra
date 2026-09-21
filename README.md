# Tensile

Dense linear algebra for .NET: blocked GEMM with hand-written micro-kernels, LU
with partial pivoting, and 1-norm condition estimation. No native dependencies,
no NuGet packages in the library itself.

It started as one question — **does RyuJIT keep a full set of vector
accumulators in registers across the k-loop, or does it spill?** — and the
answer turned out to be good enough to build on. The measurements behind that
are still below, because they are the reason any of this is worth doing in
managed code.

```csharp
using Tensile;

var a = Matrix.FromRows(new[,] { { 4.0, 1.0 }, { 1.0, 3.0 } });
var b = Matrix.FromColumnMajor<double>(2, 1, [1.0, 2.0]);

Matrix<double> x = a.Solve(b);

LuDecomposition lu = a.FactorLu();
double rcond = lu.ReciprocalCondition();
```

**[Full API guide](docs/api.md)** — matrices and views, structure-typed
dispatch, factorizations, norms, workspaces, and matrix-free operators.

## Layout

| Path | Contents |
|---|---|
| `src/Tensile` | The public assembly: compiled with unsafe code disallowed and integer overflow checking on. Matrices, views, structures, factorizations, operators. |
| `src/Tensile.Kernels` | The kernel assembly: micro-kernels, packing, GEMM, LU, triangular solves. All `internal`, all the library's unsafe code, reached through one pinning seam. |
| `src/Tensile.Interop.Blis` | The optional native BLIS binding, a separate package. The only native code loading in the family; the core never references it. |
| `tests/Tensile.Tests` | xunit suite. Contracts generic over the micro-kernel run once per supported kernel. |
| `tests/Tensile.Fuzz` | SharpFuzz harness over the public surface, run nightly under afl++. |
| `bench/Tensile.Benchmarks` | BenchmarkDotNet: GEMM, kernel ceiling, LU block-size sweep. |
| `tools/Tensile.Diagnostics` | `tensile-diag`: host ISA, BLIS dispatch, estimator accuracy. Also the single process the codegen gate drives. |
| `disasm.sh` | Dumps micro-kernel codegen and fails on accumulator spills. |

```
dotnet test Tensile.slnx -c Release                         # the unit suite
dotnet run -c Release --project tools/Tensile.Diagnostics    # what this host supports
dotnet run -c Release --project bench/Tensile.Benchmarks -- --filter '*Gemm*'
./disasm.sh                                                  # codegen gate
```

Column-major throughout, unit row stride, double precision. `Matrix<T>` is
generic over `unmanaged, INumberBase<T>` so storage and views already work for
any numeric type; the arithmetic is `double`-only and adding a type is additive.

The project targets `net10.0` and requires the .NET 10 SDK. The verification
results below were originally collected on .NET 8.

## Native BLIS comparison

The binding lives in its own package, `Tensile.Interop.Blis`, which the
benchmarks and `tensile-diag` reference and the library does not. A project
that only uses `Tensile` carries no code that loads a native library. Read
`src/Tensile.Interop.Blis/README.md` before pointing `TENSILE_BLIS_LIBRARY` at
anything on a machine you do not control.

On Debian/Ubuntu/Pop!_OS, install the shared library and development symlink:

```bash
sudo apt install libblis-dev
dotnet run -c Release
```

The loader searches for `libblis.so` or `libblis.so.4` on Linux,
`libblis.dylib` on macOS, and `blis.dll` or `libblis.dll` on Windows.
For a custom build or another library name, specify the shared library explicitly:

```bash
TENSILE_BLIS_LIBRARY=/absolute/path/to/libblis.so \
  dotnet run -c Release --project tools/Tensile.Diagnostics
```

An explicit override is authoritative: it does not fall back to the system BLIS.
Missing libraries, required exports, or unsupported integer ABIs produce a skip
message, and the managed benchmarks still run. The binding detects BLIS's native
32- or 64-bit integer ABI, independently of its BLAS compatibility integer size.
It calls `bli_dgemm` with no transposes, row strides of one, and the supplied
column strides. No C shim or NuGet dependency is needed.

BLIS runs first and reports its version, integer width, architecture selected by
`bli_arch_query_id`/`bli_arch_string`, native double GEMM micro-kernel
classification from `bli_info_get_gemm_ukr_impl_string`, and thread count.
Architecture and kernel-query enums use C `int`, not BLIS's dimension integer
type. On this machine BLIS 0.9.0 reports `haswell`; the library name and version
alone do not establish this. A `generic` architecture, unknown dispatch, or a
non-optimized kernel classification produces a warning. Do not treat such a run
as a comparison against optimized BLIS: rebuild BLIS from source with
`./configure auto`, then `make -j`, and point `TENSILE_BLIS_LIBRARY` at the built
shared library. Recheck the reported dispatch before interpreting ratios.

These queries identify the selected architecture and registered native packed
GEMM micro-kernel, not a trace of every executed kernel symbol. BLIS can choose
separate small/unpacked GEMM paths depending on matrix shape.

The harness calls `bli_thread_set_num_threads(1)` and checks the result,
overriding thread-count environment settings for a single-threaded comparison.
Both BLIS and managed kernels use the same seeded inputs, sizes, two warmup
calls, and **31 timed repetitions per size**. The table reports median latency,
latency IQR (Q3 minus Q1), and sample count. Quartiles use linear interpolation
at index `(n - 1) * p` in sorted samples for `p = 0.25, 0.5, 0.75`.
GFLOP/s is computed from median latency, never the fastest sample. Allocation
is outside the timed region; GEMM packing and native call overhead are included.
BLIS's internal packing buffers may be reused, just as managed scratch buffers
are reused. The naive baseline uses the same timing routine; the micro-kernel
ceiling uses 31 batches of 200,000 calls with median and IQR batch latency.

Managed results include `% of BLIS`: median-derived managed GFLOP/s divided by
median-derived native BLIS GFLOP/s at the same size, times 100. A value of 100
means parity; larger means the managed implementation was faster in that run.
`% of ceiling` uses the managed micro-kernel's median-derived L1-resident
ceiling. IQR describes within-run variation, not a confidence interval on the
ratio. Repeat pinned runs and examine drift before making a small percentage
performance claim. Backends still run sequentially, so frequency, thermal,
and ordering effects are not eliminated by using medians.

### Pin the benchmark

On Linux, restrict the process before .NET starts so its threads inherit a
single logical CPU. Select an allowed performance core from the local topology;
do not infer the CPU model or core type from logical-core count alone:

```bash
lscpu -e=CPU,CORE,SOCKET,MAXMHZ,ONLINE
taskset -pc $$
taskset -c 0 dotnet run -c Release --project bench/Tensile.Benchmarks -- --filter '*Gemm*'
```

CPU 0 is an allowed, SMT-enabled higher-clocked core on the verification machine.
Use another CPU number if your topology or container CPU allocation differs.
The harness reports its affinity mask on Linux/Windows and warns when it cannot
confirm a single-CPU pin; it does not silently change affinity. An unpinned run
still works but is not suitable for a close performance comparison on a hybrid
CPU. Pinning prevents P/E-core migration, not frequency drift, interrupts, or
contention from the sibling SMT thread. Keep background load low and record
the printed runtime/GC settings: constraining .NET to one CPU can also change
its GC mode. A full 31-sample run takes longer, especially for the scalar backend.

Run correctness checks without benchmarks:

```bash
taskset -c 0 dotnet run -c Release --project tools/Tensile.Diagnostics
```

Each supported backend is checked against the naive oracle on ragged and empty
shapes, packed and padded leading dimensions, and several alpha/beta values.
The checks also cover zero-beta output containing NaNs and untouched output
padding. Correctness failures set a nonzero exit code and suppress that backend's
timings. An unavailable optional backend is skipped, not treated as a failure.
Deterministic timing-statistics checks also cover odd/even sample counts,
singleton and constant samples, an outlier, and a full 31-sample sequence.

## Results from a verification run

Historical output below uses the original best-of-N method and unrecorded CPU
affinity. It predates the median/IQR and dispatch checks and must not be used to
support a close managed-versus-BLIS performance claim.

Single vCPU, virtualised Intel Xeon @ 2.1 GHz nominal (AVX-512 capable),
.NET 8.0.31, single-threaded. **Treat the absolute numbers as indicative only** —
this was a shared VM, and there was no OpenBLAS or BLIS available to compare
against. The ratios are the interesting part.

```
=== AVX-512 16x8 (MR=16, NR=8) ===
  correctness   : PASS  (worst relative residual 3.180E-016)
  kernel ceiling: 85.75 GFLOP/s (L1-resident panels, no packing)
     size        time (ms)      GFLOP/s   % of ceiling
    128            0.08        50.23          58.6
    256            0.57        58.56          68.3
    512            4.18        64.21          74.9
   1024           31.66        67.83          79.1
   2048          239.63        71.69          83.6

=== AVX2/FMA 8x6 (MR=8, NR=6) ===
  kernel ceiling: 45.18 GFLOP/s
   1024           49.52        43.36          96.0

=== scalar 4x4 (MR=4, NR=4) ===
  kernel ceiling:  8.69 GFLOP/s
   1024          239.66         8.96         103.1

=== naive triple loop ===
    256           19.98         1.68
```

Sanity check on the ceilings: 85.75 GFLOP/s at 2 FMA/cycle x 8 lanes x 2 flops
implies ~2.68 GHz; the AVX2 figure of 45.18 implies ~2.82 GHz. Those are
consistent with each other and with AVX-512 license downclocking, which means
**both micro-kernels are running at essentially the hardware's FMA issue
limit.** The full GEMM then reaches 84% (AVX-512) to 96% (AVX2) of that.

## Finding 1: no spills — the generated inner loop is what you'd hand-write

The AVX-512 k-loop, verbatim from `DOTNET_JitDisasm`:

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

Instruction mix for one k iteration: 16 `vfmadd231pd`, 8 `vbroadcastsd`
(memory-operand form, no separate load), 2 `vmovups`, 4 scalar ops of loop
overhead. Zero stack traffic. All 16 accumulators stayed in `zmm1`–`zmm16`,
`vfmadd231pd` is the accumulate-in-place form, and the broadcasts fold their
memory operand.

This is the result the experiment was built to get: **RyuJIT is not the
bottleneck for a BLIS micro-kernel.**

## Finding 2: RyuJIT's allocator is shape-sensitive, and it fails quietly

An earlier version of the probe used 16 independent FMA chains in a flat loop
with a loop-invariant constant. Same register count, same instruction set — and
RyuJIT spilled every accumulator:

```
       vmovups  zmm2, zmmword ptr [rbp-0xB0]
       vfmadd213pd zmm2, zmm0, zmmword ptr [reloc @RWD00]
       vmovups  zmmword ptr [rbp-0xB0], zmm2
```

A load/FMA/store round trip per accumulator per iteration, roughly 5x off peak.
Nothing warned about it; it just ran slowly.

The lesson for a real library: **register residency is not something you can
assume from the source.** Every hot kernel needs its disassembly checked, and
that check belongs in CI, not in someone's memory. Note also that spills here
were `rbp`-relative, not `rsp`-relative, so grep for both.

## Finding 3: tiered compilation will lie to you

With default tiering and no `AggressiveOptimization`, the first measured run
reported 14–16 GFLOP/s for sizes 128–1024 and 73 for 2048. Large GEMMs make few
calls into the kernel, so it never tiers up within the measurement window.

Both `Gemm.Multiply` and every `Execute` are annotated
`[MethodImpl(MethodImplOptions.AggressiveOptimization)]`, which fixes it. When
benchmarking by hand, cross-check with `DOTNET_TieredCompilation=0`.

## Reading the disassembly yourself

```bash
./disasm.sh kernel.asm
```

It dumps the kernels, reports FMA and spill counts per kernel, and exits
non-zero if any accumulator reached the stack. CI runs the same script.

Two traps it exists to avoid. First, do not dump through `dotnet run`: the SDK
driver and the application are separate processes and both honour
`DOTNET_JitStdOutFile`, so they overwrite each other's output and kernels go
missing from the dump without any error. Run the built binary directly. Second,
scope the grep to the kernel listings — plenty of framework methods are also
called `Execute`, and a spill in one of those is not your problem.

What you want to see in the loop body: MR/8 * NR `vfmadd231pd`, NR
`vbroadcastsd`, MR/8 `vmovups` loads, and nothing else but the loop counter. On
.NET 10.0.112 the totals including the write-back are 32 `vfmadd` for the
AVX-512 16x8 kernel and 24 for the AVX2 8x6, with zero spills.

## What this does not answer

- **No OpenBLAS comparison.** BLIS is now an optional baseline in the harness;
  the historical .NET 8 results above predate that integration.
- **Block sizes are untuned.** `GemmScratch.For<T>` still uses MC=288, KC=384,
  NC=4096 as placeholders; `ParallelGemmScratch` uses cache-derived KC=256,
  MC=144, which measured better below n=2048 and has not been ported to the
  serial path. Sweep them, or derive them per the BLIS analytical model (Low et
  al., TOMS 2016).
- **Double only, no transposes, no complex.** Deliberately.
- **One machine, virtualised, one core.** Re-run on your own hardware before
  drawing conclusions.

## Suggested next steps

1. Re-run on your hardware; confirm the no-spill result there.
2. Compare the managed kernels with BLIS on the same machine and record its build/version.
3. Sweep MC/KC/NC, and port the cache-derived values to the serial path.
4. Try MR=24 NR=8 (24 accumulators, better FMA-to-load ratio, 28 of 32 zmm
   live) and see where RyuJIT's allocator gives out.
5. Only then decide managed vs native.
