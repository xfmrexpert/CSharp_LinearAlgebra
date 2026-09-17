# Tensile: secure-by-design proposal

Status: **approved** — every recommendation in §11 accepted. Phases 1–6 of
§10 have landed; Phase 7 (re-measure, consumer-facing `docs/security.md`) is
next. New functionality (Cholesky, `expm`, complex)
stays paused until §10 is complete.

This document says what "secure" means for a dense linear algebra library,
which guarantees Tensile will make, how the design makes each guarantee hold by
construction rather than by vigilance, and how we will know.

---

## 1. Why this is being done now

Two memory-safety bugs were found in the first review of the public API — a
`Slice` bounds check that overflowed `int` and handed back a view 16 GB past its
buffer, and a `CopyTo` that corrupted its own source on overlapping windows. An
audit immediately afterwards found a third of the same class (`n * t` wrapping
into an allocation size in `NormEstimate`).

Three in one week is not bad luck. It is what happens when the invariant "this
view is in bounds" is re-derived by hand at every call site, in a codebase where
every file is `unsafe` and a mistake is a write primitive rather than a wrong
answer. Point-fixing them would leave the mechanism that produced them intact.

The reasonable time to change the mechanism is now: pre-1.0, no external
consumers, and an API that is days old.

---

## 2. Threat model

### In scope

- **T1 — Attacker-influenced dimensions and values.** A consumer puts Tensile
  behind something that accepts matrices from outside — a solver service, a
  file-fed FEM model. Rows, columns, strides and element values become attacker
  data. This is the primary threat and everything below is shaped by it.
- **T2 — Memory unsafety as the consequence of T1.** The library is the
  component that turns a bad integer into a corrupted heap. In a managed
  library a bounds bug yields a wrong answer; here it yields a read/write
  primitive.
- **T3 — Supply chain.** Build reproducibility, dependency pinning, CI token
  scope, mutable action tags.
- **T4 — Resource exhaustion.** Unbounded allocation from attacker-chosen
  dimensions. Real but weak: any compute library can be asked to do too much
  work, and the fix is a policy limit rather than a design change.

### Explicitly out of scope

Stated so nobody later assumes otherwise.

- **Confidentiality of matrix contents.** There is no cryptography here and
  no secret data by design. Pivot selection is data-dependent and `Axpy`
  skips zero multipliers, so run time leaks matrix structure. That is a
  non-goal, not a bug.
- **Network, file, serialization, authentication.** None exist in the library.
- **An adversary already executing code in the process.** If they control your
  environment variables or your heap, no library design saves you. We do,
  however, refuse to *widen* that blast radius — see I8.
- **Numerical robustness** (NaN propagation, ill-conditioning, overflow to
  infinity). Correctness concerns, tracked separately, and not smuggled in
  under the security heading.

---

## 3. The design principle

The library's value is an inner loop with no bounds checks, raw pointers and
aligned memory. "Secure by design" cannot mean removing that; it would remove
the reason the project exists.

It means this instead:

> **Establish every invariant once, at a boundary the type system enforces,
> and make the region below that boundary small, enumerable and unable to
> receive an invalid input.**

Concretely: validate at construction and never again; carry proof of validity
in the type; let the runtime, not our arithmetic, be the last line of defence;
and confine `unsafe` to an assembly the compiler can certify the public API
does not contain.

The test of the design is not "did we check carefully" but **"when we get the
arithmetic wrong, what happens?"** Today the answer is memory corruption. After
this, the answer is an exception.

---

## 4. Invariants

These are the guarantees. Each has a number, an enforcement mechanism, and a
verification in §8. "Enforced by" is the important column: an invariant held by
the compiler or by construction is categorically stronger than one held by tests.

| # | Invariant | Enforced by |
|---|---|---|
| **I1** | Every memory access through a view lies within the buffer the view was bound to. | Runtime (`Span<T>` slicing). Our arithmetic is checked *in addition*, never *instead*. |
| **I2** | A `MatrixShape` cannot exist unless rows ≥ 0, columns ≥ 0, stride ≥ rows, and its required extent fits in `int`. | Construction; the only constructor validates. |
| **I3** | No public type accepts, stores or returns a raw pointer. | Reflection test over the public surface; compiler (see I4). |
| **I4** | The public assembly contains no `unsafe` code. | **Compiler.** `AllowUnsafeBlocks=false` on `Tensile`; kernels live in a separate assembly. |
| **I5** | Every public entry point either produces the documented result or throws. It never partially writes, reads out of bounds, or leaks a buffer. | Exception-safety discipline; property tests. |
| **I6** | Storage cannot be freed while any view of it is reachable. | **Construction.** GC-tracked pinned storage; a `Span` over it is a tracked root. |
| **I7** | No size, offset or extent computation can silently wrap. | **Compiler** (`CheckForOverflowUnderflow` in the safe assembly) plus one audited helper for the kernel assembly. |
| **I8** | The core assembly loads no native code. | Reflection test (no `DllImport` / `NativeLibrary` reference); interop is a separate opt-in package. |
| **I9** | Allocation goes through one path with a configurable ceiling. | Single allocator; policy object. |

---

## 5. Design, type by type

### 5.1 `MatrixShape` — validity as a value

A `readonly record struct` holding `Rows`, `Columns`, `Stride`, constructible
only through a validating constructor. Every piece of shape arithmetic lives
here and is checked.

```csharp
public readonly record struct MatrixShape
{
    public MatrixShape(int rows, int columns, int stride = 0);   // validates or throws

    public int Rows { get; }
    public int Columns { get; }
    public int Stride { get; }          // >= Rows; == Rows when packed

    /// Elements a buffer needs: (Columns - 1) * Stride + Rows, or 0. Computed
    /// in long and required to fit int, so it can be compared to Span.Length.
    public int RequiredExtent { get; }

    public int OffsetOf(int row, int column);                        // checked
    public MatrixShape Sub(int row, int column, int rows, int columns); // checked, same stride
    public bool IsEmpty, IsSquare, IsContiguous;
}
```

Why a separate type: today `MatrixView` fuses three concerns — shape, ownership
and access — and its constructor takes shape and pointer from the caller with
no way to check they agree. Splitting shape out means "is this shape valid" is
answered once, and "does this shape fit this buffer" becomes a single
comparison at bind time (§5.3).

`RequiredExtent` is `(Columns-1)*Stride + Rows`, not `Stride*Columns`: the last
column needs no trailing padding, and using the tight bound means a caller's
exactly-sized buffer binds.

### 5.2 `Matrix<T>` — storage without a lifetime problem

**Recommendation: GC-pinned managed storage, not native memory.**

```csharp
public sealed class Matrix<T> where T : unmanaged, INumberBase<T>
{
    private readonly T[] _storage;   // GC.AllocateUninitializedArray(n + pad, pinned: true)
    private readonly int _offset;    // so that &_storage[_offset] is 64-byte aligned

    public MatrixShape Shape { get; }
    public MatrixView<T> View => MatrixView<T>.Bind(_storage.AsSpan(_offset, Shape.RequiredExtent), Shape);
    // no Dispose, no finalizer
}
```

What this buys, and why it is the secure-by-design choice rather than a
convenience:

- **I6 by construction.** A `Span<T>` over a managed array is a GC-tracked
  reference. While any `MatrixView` is live on any thread's stack, the array
  cannot be collected. Use-after-free is not *prevented*; it is *unexpressible*.
  Today, disposing a `Matrix` on one thread while another is mid-GEMM through a
  view is a use-after-free, and no bounds check anywhere touches that.
- **No `IDisposable`, no finalizer, no double-free, no leak-on-forgotten-dispose.**
  An entire bug family leaves the core.
- **`Matrix<T>` becomes safe code.** It needs no pointer at all.

Costs, stated honestly:

- **Alignment is manual.** Pinned-object-heap arrays are not guaranteed
  64-byte aligned, so over-allocate by one cache line and start the view at the
  aligned offset. The kernels already use unaligned load instructions
  (`vmovups`, confirmed in the disassembly), so alignment is a cache-line
  nicety rather than a correctness requirement; we keep it because the
  measurements were taken with it.
- **Pinned arrays never move**, so heavy churn of many small matrices could
  fragment the heap. For a solver holding a handful of large matrices this is
  what you want (it behaves like the LOH). For a hot loop creating thousands
  of tiny matrices it is the wrong tool, and `Workspace` reuse is the answer
  there. This should be documented rather than hidden.
- **`Span<T>` is capped at `int.MaxValue` elements**, so a matrix is capped at
  ~2.1×10⁹ elements (16 GB of `double`). `Matrix<T>` is *already* capped there
  today, by an incidental `checked((int)count)`; this makes it a documented
  contract. A 1000×1000 target is four orders of magnitude below it. Larger
  would need segmented storage, which is a different design.
- **The performance model changes** — same instructions, same alignment, but
  different allocator. It must be re-measured (§9), not assumed.

The alternative — keep native memory, keep `IDisposable`, and either document
"disposal during use is undefined" or add reference counting — is workable but
leaves I6 as a rule callers must follow. Reference counting with `ref struct`
views would need a `using var lease = matrix.Lease()` ceremony because ref
structs have no scope-exit hook. Recommendation is POH; it is the only option
that makes I6 true without asking anything of the caller.

### 5.3 `MatrixView<T>` / `ReadOnlyMatrixView<T>` — spans, not pointers

```csharp
public readonly ref struct MatrixView<T> where T : unmanaged
{
    private readonly Span<T> _buffer;      // Length >= Shape.RequiredExtent, checked at Bind
    public MatrixShape Shape { get; }

    // The ONLY ways to obtain a view:
    public static MatrixView<T> Bind(Span<T> buffer, MatrixShape shape);   // checks extent, or throws
    public MatrixView<T> Slice(int row, int column, int rows, int columns); // derived; see below

    public Span<T> Column(int j) => _buffer.Slice(Shape.OffsetOf(0, j), Shape.Rows);
    public ref T this[int i, int j] => ref _buffer[Shape.OffsetOf(i, j)];
    public void CopyTo(MatrixView<T> destination);   // overlap-safe, as now
}
```

**The public pointer constructor is removed.** This is the single most
important change in the document. A caller who genuinely has a `T*` writes
`new Span<T>(p, length)` themselves — their `unsafe` block, correctly
attributed, and it forces them to state the length, which is precisely the
fact the library needs and today never receives.

`Slice` derives the sub-shape with checked arithmetic *and* re-slices the
buffer, so it is defended twice: our arithmetic and the runtime's. If our
arithmetic is wrong, `Span.Slice` throws. That is I1.

Binding accepts a span over *any* memory — pinned, native, or an ordinary
managed array. The last case is what makes the pinning seam (§5.5) necessary.

### 5.4 Structures, factorizations, operations

Unchanged in shape; changed in plumbing. `StructuredMatrix<T, TStructure>` wraps
a view. `ITriangularStructure.SolveInPlace` takes views. `LuDecomposition` owns
a `Matrix<double>` and loses its `IDisposable`. `Workspace` keeps its lock and
its `Shared` instance.

**`ILinearOperator` changes signature.** It is a public extension point — it is
how `expm` and any matrix-free operator plug in — and today it takes raw
pointers. It becomes:

```csharp
public interface ILinearOperator
{
    int Order { get; }
    void Apply(ReadOnlyMatrixView<double> x, MatrixView<double> y);
    void ApplyTranspose(ReadOnlyMatrixView<double> x, MatrixView<double> y);
}
```

An implementer can no longer hand the estimator an `Order` that disagrees with
what its buffers can hold: the views it receives are already bound and checked.
The `n * t` overflow that motivated this document lands in the estimator's own
allocation path, which is fixed by I7 and I9.

### 5.5 The pinning seam — where spans become pointers

The kernel assembly (§5.6) has exactly one place that turns a span into a
pointer, and it does so with `fixed`. It cannot take the view types — they are
defined in the public assembly, which references the kernel assembly and not
the other way round — so it takes its own minimal equivalent, `Operand` /
`Target`: a span plus rows, columns and stride, length-checked on construction.
The public assembly repackages a view into one in a single safe method
(`ViewOperands`) that contains no arithmetic.

```csharp
internal static unsafe class KernelEntry
{
    public static void Multiply<TKernel>(GemmDispatch dispatch, Operand a, Operand b, Target c,
                                         double alpha, double beta)
        where TKernel : struct, IMicroKernel
    {
        Require(a.Columns == b.Rows, "inner dimensions");
        Require(c.Rows == a.Rows && c.Columns == b.Columns, "destination shape");

        fixed (double* pa = a.Data)
        fixed (double* pb = b.Data)
        fixed (double* pc = c.Data)
        {
            dispatch.Multiply<TKernel>(a.Rows, b.Columns, a.Columns,
                alpha, pa, a.Stride, pb, b.Stride, beta, pc, c.Stride);
        }
    }
}
```

Nothing returned from the seam holds a pointer. `LuFactorization` carries the
pivots (a managed array) and the diagnostics; the solves take the factors back
as an operand. That is what lets `Workspace.FactorLu` be public again: it
factors a `Matrix<double>` in place, and the decomposition it returns shares
that matrix's GC-owned storage rather than pointing into it.

Two properties matter:

- **`fixed` pins if the span is over movable managed memory and is a no-op if
  it is not.** So binding an ordinary `double[]` is safe: the GC cannot move it
  mid-kernel. This is why `Bind` can accept any span.
- **No pointer escapes the `fixed` scope.** The parallel GEMM captures pointers
  into `Parallel.For` bodies as `nint`; that is fine *only* because
  `Parallel.For` is synchronous and completes inside the scope. This becomes a
  stated contract of the seam: every operation below it is synchronous, and any
  future asynchronous path must pin differently. This constraint is the price
  of accepting movable memory, and it is worth paying.

Every kernel-assembly entry point documents its preconditions on shape and
packing layout, asserted under `Debug` (§8, "debug assertions").

### 5.6 Two assemblies — so the compiler certifies the boundary

| Assembly | `AllowUnsafeBlocks` | Visibility | Contents |
|---|---|---|---|
| `Tensile` | **false**, and `CheckForOverflowUnderflow` **true** | public API | `Matrix`, `MatrixShape`, views, structures, `LuDecomposition`, `Workspace`, operations, `ILinearOperator`, the `normest1` driver |
| `Tensile.Kernels` | true | **all `internal`**; `InternalsVisibleTo` → `Tensile`, tests, bench, diagnostics | micro-kernels, packing, GEMM drivers, LU, triangular solves, Blas1/2, exact norms, `KernelEntry` |
| `Tensile.Interop.Blis` | true | public, **separate package** | the BLIS binding |

The reason for the split is not that a consumer cannot reference
`Tensile.Kernels.dll` — they can, and a previous draft of this argument
overstated that. It is that `AllowUnsafeBlocks` is a per-project switch, so this
is the *only* way to have the compiler prove the public assembly contains zero
unsafe code. That is I4, held by the compiler rather than by review. With every
kernel type `internal`, referencing the DLL gains a consumer nothing short of
reflection.

`Tensile.Kernels` ships inside the same NuGet package as `Tensile`.

The `normest1` driver landed on the safe side of the line, not the kernel side
the first draft of this table put it on. It is bookkeeping — sign matrices,
column sums, a sort — over O(n·t) buffers, and every product it needs goes
through an `ILinearOperator`, so nothing in it wants a pointer. Writing it over
managed arrays under overflow checking cost nothing measurable and removed a
few hundred lines of unsafe code. The O(n²·t) products it asks for are still
the kernel assembly's.

The compiler half of I4 turned out to be observable after all: the C# compiler
marks a module containing unsafe code with `UnverifiableCodeAttribute`. The
surface test asserts the mark is present on `Tensile.Kernels` (so the detector
is known to fire) and absent on `Tensile`.

**The "opt-in loophole" is dropped.** The legitimate advanced need — zero-copy
against memory the caller owns — is met by `Bind` over their span (§5.3), not
by exposing pointer entry points. What is lost is an O(1) validation per call
against O(n²) minimum work, and the hackable-stack value in `CLAUDE.md` is
preserved for the project's own tests, benchmarks and diagnostics through
`InternalsVisibleTo`.

### 5.7 Interop — out of the core

`Tensile.Interop.Blis` becomes its own project and package. The core assembly
never calls `NativeLibrary.Load`. Today every consumer of Tensile carries a code
path that `dlopen`s a path read from `TENSILE_BLIS_LIBRARY`; after this, only a
consumer who explicitly references the interop package does. That is I8.

The environment variable stays, in the interop package, as a benchmarking
convenience; its README states plainly that it loads and executes native code
from an attacker-controllable location and is for development machines only.

### 5.8 Arithmetic — checked by the compiler where it is cheap

**Recommendation: `<CheckForOverflowUnderflow>true</CheckForOverflowUnderflow>`
on `Tensile`.** Everything not in the kernel assembly is then overflow-checked
by the compiler, with no per-site discipline. The safe assembly has no hot
loops — the O(n²) work it does (norms, structure checks) is memory-bound and a
checked add is free against a cache miss. The kernel assembly stays unchecked
and does its index arithmetic in `nint` through one audited helper.

This is the difference between "we remembered to write `checked`" and "we could
not have forgotten". It would have caught the `Slice` overflow at the moment of
the bug rather than in review.

### 5.9 Allocation — one path, one ceiling

All storage allocation goes through a single internal allocator that enforces a
policy limit and is exception-safe. The limit is a static, process-wide
`TensileLimits.MaxElements` defaulting to the span cap, settable lower by a
consumer that wants a service to refuse a 16 GB request rather than attempt it.

This is a policy control, not a memory-safety one — an over-large allocation
already fails cleanly — but it is what turns T4 from "the runtime will
eventually throw `OutOfMemoryException`" into "the request is refused at the
door with a clear message".

As built: the path is `Storage` in the public assembly, with `Pinned<T>` for
matrix storage and `Array<T>` for work; the only other `new T[]` in that
assembly is inside `Storage` itself. The limit is measured against the
caller's request (a matrix's required extent, a probe panel's n·t), not
against the allocation after alignment padding, and per allocation rather than
in total: an operation that allocates five panels is bounded five times, not
once. Refusal is `AllocationLimitException` with `Requested` and `Limit`, a
type of its own so a service can map it to "too large" and keep
`OutOfMemoryException` for "in trouble". The kernel assembly allocates only
work bounded by operands the caller already holds — a pivot array of
min(m, n), a row-sum vector of m, packing buffers of block constants times m —
and sits outside the policy by design.

### 5.10 Concurrency

`Workspace` keeps its internal lock; the argument for it is unchanged. The one
new statement: **a `Matrix<T>` is safe to read concurrently and is not safe to
write concurrently with any other access**, which is the same contract as a
`T[]`. With POH storage there is no disposal to race against.

---

## 6. What changes for a caller

All breaking, all deliberate.

| Today | After |
|---|---|
| `new MatrixView<double>(ptr, rows, cols, stride)` | `MatrixView<double>.Bind(span, new MatrixShape(rows, cols, stride))` |
| `using var a = new Matrix<double>(m, n);` | `var a = new Matrix<double>(m, n);` — no `using` |
| `using LuDecomposition lu = a.FactorLu();` | `LuDecomposition lu = a.FactorLu();` |
| `Tensile.Primitives.Gemm.Multiply<K>(m, n, k, alpha, pa, lda, ...)` | not available; use `Workspace.Multiply(viewA, viewB, viewC)` |
| `ILinearOperator.Apply(int t, double* x, int ldx, double* y, int ldy)` | `Apply(ReadOnlyMatrixView<double> x, MatrixView<double> y)` |
| `Tensile.Interop.Blis` in the core package | separate `Tensile.Interop.Blis` package |

Everything in the ergonomic layer that took a `Matrix<double>` keeps working;
the fluent surface (`Multiply`, `Solve`, `FactorLu`, norms, structured solves)
is unchanged in name and intent.

---

## 7. What does *not* change

- The micro-kernels, byte for byte. The codegen gate (`disasm.sh`) must report
  the same 32 / 24 `vfmadd` counts and zero spills after the move.
- Packing, the five-loop GEMM driver, `ParallelGemm`, LU, the triangular
  solves, `normest1`. They move assemblies and lose `public`; their bodies are
  untouched.
- The structure vocabulary and the compile-time dispatch through it.
- The benchmark and diagnostics projects, beyond `InternalsVisibleTo` and
  referencing the interop package.

---

## 8. Verification — mapped to the invariants

All four layers agreed on, plus one that falls out of the design.

| Layer | What it checks | Invariants |
|---|---|---|
| **Compiler** | `AllowUnsafeBlocks=false`; `CheckForOverflowUnderflow=true`; kernel types `internal`. Fails the build, not a test. | I4, I7 |
| **Public-surface reflection test** | No public type or member in `Tensile` has a pointer type in its signature; the assembly has no `DllImport` and no reference to `NativeLibrary`; no public type implements `IDisposable` over native memory. Runs in the unit suite, so the guarantee cannot regress silently. | I3, I8 |
| **Property tests over hostile inputs** | For every public entry point: `int.MaxValue`, `int.MinValue`, `-1`, `0`, and pairs whose product wraps. Assert: succeeds or throws the `ArgumentException` family, never anything else — an `OverflowException` or `OutOfMemoryException` means the input got past validation; no partial write (sentinel-filled destination unchanged on throw). **Written first, against the invariants, before the refactor** — several fail on today's code, and turning them green is the migration. Two rules bound what can be in the suite: nothing may depend on the host's memory (shapes in the tens-of-gigabytes range behave differently under overcommit and are excluded), and nothing may corrupt memory on today's code (a test that overflows the heap is a crashed host, not a red test — so the clean-failing form of a defect is exercised and its corrupting form documented). | I1, I2, I5, I7, I9 |
| **Continuous fuzzing** | SharpFuzz over `MatrixShape` construction, `Bind`, `Slice`, and the ergonomic entry points, run on a schedule (not per-PR), with the property-test cases as the seed corpus. Finds the pairs nobody enumerated. | I1, I2, I5, I7 |
| **Static analysis** | CodeQL on a schedule; .NET analyzers at `AnalysisLevel=latest-all` with the unsafe-code rules as errors in the safe assembly. Weak on the arithmetic class that bit us, so it is a floor rather than the guarantee. | I3, I4 |
| **Debug assertions** | Kernel-assembly entry points assert their preconditions (shape agreement, packed-panel extents, alignment) under `Debug`, compiled out of `Release`. Documents the contract executably. | I1 (kernel side) |
| **By construction** | I6 has no test because it has no failure mode: there is no `Dispose` to race. This is the point. | I6 |

The ordering matters: **the property tests come first.** They are the
specification. They are red against `main` today, and each phase of §10
turns some of them green.

One limit, stated plainly: a test can only be written against a surface that
compiles. Invariants about *behaviour of the existing API* and about the
*shape of the public surface* (reflection) can be committed red now. Behavioural
tests for types that do not yet exist — `MatrixShape`, `Bind`, `TensileLimits`
— cannot; they arrive with those types in Phases 2 and 5, red-first within each
phase. The reflection tests bridge the gap where they can (e.g. "`Matrix<T>` is
not `IDisposable`" pins the Phase 2 storage decision today).

---

## 9. Performance guardrails

The design is only acceptable if the hot path does not pay for it. Claims in
this document about where bounds checks land are reasoning, not measurement,
and `CLAUDE.md` is explicit that reasoning about performance has been wrong
before.

**Where checks land, by construction:**

| Path | Bounds checks | Work |
|---|---|---|
| Per-column slice in packing | O(n) per slab | O(n²) per slab |
| Per-micro-tile pointer in the GEMM driver | ~2.6×10⁵ at n=2048 | 1.7×10¹⁰ flops |
| Inside the k-loop | **zero** — pointer minted once per kernel call via `fixed` | O(n³) |

**Measurement plan**, on the 12700H, before and after, using the existing
BenchmarkDotNet suite:

- Kernel ceiling: must be identical within run-to-run noise. The kernel is
  byte-identical; any difference is a measurement artefact and should be
  chased as one.
- GEMM at n=128: the most sensitive case, where overhead is largest relative
  to work. Acceptance: within 3% of before.
- GEMM at n=2048: within 1%.
- LU at n=1024, nb=32: within 2%.
- `disasm.sh`: identical FMA counts, zero spills.

If n=128 regresses beyond 3%, the first suspect is `fixed` on POH memory
(should be free; verify), the second is `CheckForOverflowUnderflow` catching a
loop we thought was cold. Either is fixable without abandoning the design.

---

## 10. Migration order

Each phase leaves the build green and the tests passing. Nothing is merged
red.

0. **Sign-off on this document**, and decisions on the open questions in §11.
1. **Property-test suite**, written against §4, committed red. *Done: 82 tests, 18 red.* This is the
   specification and the acceptance criterion for everything after.
2. **`MatrixShape`; `Matrix<T>` on POH; views on `Span<T>`; pointer
   constructor removed.** *Done: 18 red → 4; I1, I2, I5, I6, I7 green.* `Tensile` is still one assembly at this point.
   Several property tests go green here (I1, I2, I6, parts of I5).
3. **Split `Tensile.Kernels`.** `AllowUnsafeBlocks=false` and
   `CheckForOverflowUnderflow=true` on `Tensile`; `KernelEntry` seam;
   primitives `internal`; `ILinearOperator` re-signatured. I3, I4, I7 go
   green; the surface reflection test lands here. *Done: 4 red → 1 (I8).
   Three things came out differently from the sketch above, all recorded in
   §5.5–5.6: the seam takes `Operand`/`Target` rather than views, because the
   view types live in the assembly above it; the `normest1` driver moved up
   into the safe assembly rather than down into the kernels; and the BLIS
   binding is parked, internal, in the kernel assembly until Phase 4, so the
   I8 test now checks both assemblies and stays honestly red.*
4. **`Tensile.Interop.Blis` out to its own package.** I8 green. *Done: the
   invariant suite is fully green. The package references `Tensile`, never
   the reverse, and a test asserts neither core assembly references it. Its
   public product takes bound views and pins inside its own seam; the
   pointer overload is internal to the benchmarks. The environment-variable
   override stays, with the README warning §5.7 asked for.*
5. **Allocator and limits.** I9 green. *Done. `Storage` is the one path;
   `TensileLimits.MaxElements` defaults to `Array.MaxLength` per §11 item 3;
   refusal is an `AllocationLimitException` carrying the request and the
   limit, thrown before any memory is asked for and deliberately unrelated to
   `OutOfMemoryException`. The limit is applied to the request, per
   allocation, before padding; shape validity is checked first, so I2 still
   answers for impossible shapes. The kernel assembly's own allocations are
   bounded by operands the caller already holds and sit outside the policy,
   as §5.9 now records.*
6. **CI hardening**: actions pinned to SHAs, `permissions: contents: read`,
   `packages.lock.json`, CodeQL job, scheduled fuzz job. *Done. Also:
   `AnalysisLevel=latest-all` on `Tensile` with five rules switched off in
   its `.editorconfig`, each with its reason; `latest-recommended` on the
   other two assemblies; Dependabot for the pins and the lock files. The
   fuzz harness's first local run found a real defect within 90 seconds
   (CLAUDE.md finding 11): a 0 × 2³¹ matrix is valid and every column walk
   took two billion steps to do nothing. That is T4 from a 16-byte input,
   and neither the property tests nor the review saw it, which is the case
   for the fuzzer made by the fuzzer.*
7. **Re-measure on the 12700H** (§9). Update `CLAUDE.md`'s measured results
   and `docs/api.md`. Write `docs/security.md` as the consumer-facing statement
   of the guarantees in §4.

Then features resume, on a foundation where the next `Slice`-class bug is an
exception rather than a CVE.

---

## 11. Open questions for sign-off

1. **POH storage (§5.2) vs. keeping native memory.** Recommendation: POH. It is
   the only option that makes I6 hold without a caller-side rule. Cost is
   re-measurement and a documented `int.MaxValue`-element cap that already
   exists in practice.
2. **`CheckForOverflowUnderflow` assembly-wide on `Tensile` (§5.8).**
   Recommendation: yes. The safe assembly has no hot loops; the guarantee is
   worth more than the nanoseconds.
3. **Default for `TensileLimits.MaxElements` (§5.9).** Recommendation: the span
   cap, i.e. no policy limit unless a consumer sets one. A library should not
   guess a service's memory budget.
4. **Keep `TENSILE_BLIS_LIBRARY` in the interop package (§5.7)?**
   Recommendation: yes, with the README warning; it is a benchmarking tool in
   an opt-in package.
5. **Fuzzing cadence and host.** Nightly on GitHub Actions is cheap and
   sufficient; anything continuous needs a dedicated runner. Recommendation:
   nightly.

---

## 12. What this does not solve

Being honest about the residue.

- **Kernel correctness is still on us.** Inside `Tensile.Kernels` the
  arithmetic is unchecked and the pointers are raw; that is the whole point.
  The design shrinks that region and prevents invalid input reaching it. It
  does not verify the kernels — the residual tests and the codegen gate do.
- **`Bind` trusts the span's length.** A caller who constructs
  `new Span<T>(p, wrongLength)` has lied to the runtime, and nothing downstream
  can detect it. That is correctly their `unsafe` block, but it is a boundary,
  not a guarantee.
- **Denial of service is bounded, not eliminated.** A consumer can still be
  asked to factor a 40,000×40,000 matrix within the limit. `MaxElements` is a
  policy knob; the policy is theirs.
- **Timing side channels are a stated non-goal** (§2), not a solved problem.
