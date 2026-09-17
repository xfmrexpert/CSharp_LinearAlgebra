# Tensile API

Dense linear algebra for .NET. Column-major, double precision, with hand-written
micro-kernels underneath.

The surface is two layers, and which one you want depends on whether allocation
matters to you.

| Layer | Namespace | For |
| --- | --- | --- |
| Ergonomic | `Tensile` | Ordinary code. Allocates results, owns its memory, checks its arguments. |
| Primitives | `Tensile.Primitives` | Inner loops. Pointers, strides, caller-supplied buffers, no validation. |
| Interop | `Tensile.Interop` | The optional BLIS binding, used as a benchmark baseline. |

Nothing in the ergonomic layer hides anything in the primitive layer: both are
public, and dropping down is a supported move rather than an escape hatch.

---

## Getting started

```csharp
using Tensile;

using var a = Matrix.FromRows(new[,]
{
    { 4.0, 1.0 },
    { 1.0, 3.0 },
});

using var b = Matrix.FromColumnMajor<double>(2, 1, [1.0, 2.0]);

using Matrix<double> x = a.Solve(b);        // LU with partial pivoting
using Matrix<double> product = a.Multiply(x);   // blocked GEMM
```

Everything that owns storage is `IDisposable`, because the storage is native
rather than a `T[]`. Dropping a reference does not corrupt anything — the
finalizer still frees — but it leaks until the next GC notices.

---

## Matrices and views

`Matrix<T>` owns 64-byte aligned native memory in column-major order. It is
generic over `unmanaged, INumberBase<T>`, so `Matrix<float>` and
`Matrix<Half>` hold data today even though the arithmetic below is
`double`-only.

```csharp
using var m = new Matrix<double>(rows: 100, columns: 40);
using var z = Matrix.Zeros<double>(8, 8);
using var i = Matrix.Identity<double>(8);
using var f = Matrix.FromColumnMajor<double>(2, 2, [1, 2, 3, 4]);
```

A **view** is a borrowed window. `MatrixView<T>` and `ReadOnlyMatrixView<T>` are
`ref struct`s, so the compiler prevents them being stored in a field, boxed, or
captured by an async method — the usual ways a borrowed pointer outlives its
owner.

```csharp
MatrixView<double> block = m.Slice(row: 10, column: 5, rows: 20, columns: 10);
block.Fill(0.0);                 // writes through to m

Span<double> column = m.Column(3);   // contiguous, and therefore free
```

There is deliberately no `Row`. In column-major storage a column is contiguous
and a row is not, so a `Row` returning a span would be a lie and one returning a
copy would hide its cost.

`Stride` may exceed `Rows`; a slice keeps its parent's stride rather than
repacking. Every operation handles a strided view, and the tests check that
padding is never written.

`CopyTo` is safe between overlapping windows — two slices of one matrix, say —
and stages through a temporary when it detects an overlap. Copying column by
column would otherwise destroy a source column before reading it, since
`Span.CopyTo` only protects each column individually.

---

## Structure in the type system

LAPACK puts the shape of a matrix in the function name — `dgesv` against
`dtrsv` against `dposv` — so choosing wrong compiles cleanly and returns a
confidently wrong answer. Here the shape is a type parameter, so the wrong
choice does not compile and the right one is selected with no run-time branch.

```csharp
using Matrix<double> u = BuildUpperTriangular();

// Back substitution. No factorization, no branch, chosen at compile time.
using Matrix<double> x = u.As<UpperTriangular>().Solve(b);
```

Shipped structures:

| Structure | Referenced part | Solve |
| --- | --- | --- |
| `General` | everything | needs a factorization |
| `UpperTriangular` | diagonal and above | back substitution |
| `LowerTriangular` | diagonal and below | forward substitution |
| `UnitLowerTriangular` | strictly below; diagonal assumed 1 | forward substitution |

### What a structure claims, and what it does not

A structure says **which part of the storage an operation reads**. It does *not*
say the rest is zero, and it cannot: LU packs `L` and `U` into one array, so the
triangle a structure ignores routinely holds the other factor.

```csharp
using LuDecomposition lu = a.FactorLu();

lu.Lower.SolveInPlace(x.View);   // reads strictly below the diagonal
lu.Upper.SolveInPlace(x.View);   // reads the diagonal and above
// Both view the same storage. Both are correct.

UpperTriangular.UnreferencedPartIsZero(lu.Upper.View);   // false — L is there
```

`As<TStructure>()` is an unchecked assertion, the same trust a BLAS call places
in its `uplo` argument. It is an instance method, so the element type comes from
the matrix and only the structure is named. When the claim is about data you did
not produce, `AsChecked<TStructure>()` verifies that the unreferenced part really
is zero — O(n²), and it rejects a packed factorization by design.

The safest structured matrices are the ones you never assert: `lu.Lower` and
`lu.Upper` have their shape by construction.

---

## Factorizations

```csharp
using LuDecomposition lu = a.FactorLu();     // a is not modified

using Matrix<double> x  = lu.Solve(b);
using Matrix<double> xt = lu.SolveTransposed(b);

bool broken   = lu.IsSingular;          // an exactly zero pivot
double rcond  = lu.ReciprocalCondition();
double det    = lu.Determinant();
```

`FactorLu` copies, so your matrix survives. When the input is already scratch,
factor in place instead and skip the copy:

```csharp
using Tensile;
using Tensile.Primitives;

// Any matrix you do not need intact afterwards; it is overwritten with the
// packed factors, and must outlive the factorization that indexes into it.
using Matrix<double> scratch = a.Clone();
using var workspace = new Workspace();

using LuFactorization factorization = workspace.FactorLu(scratch.View);
```

Three things worth knowing:

- **`IsSingular` is narrower than "singular".** It means an *exactly* zero
  pivot. A duplicated column is mathematically singular but its pivot comes out
  as rounding noise, so the factorization completes and `IsSingular` stays
  false — identical to `dgetrf`. Ask `ReciprocalCondition` about numerical
  singularity.
- **`PivotRatio` is not a condition number.** It is a cheap trouble indicator
  and can be optimistic by orders of magnitude.
- **`Determinant` overflows.** The product of n numbers has roughly n times the
  exponent range of one. It is for small problems and tests; it is a poor way to
  ask whether a matrix is invertible.

---

## Norms and conditioning

```csharp
double one   = a.OneNorm();          // exact, O(mn)
double inf   = a.InfinityNorm();     // exact, O(mn)
double frob  = a.FrobeniusNorm();    // exact, O(mn)

double estimate = a.EstimateOneNorm(power: 3);   // ||A^3||_1, without forming A^3
```

`EstimateOneNorm` is Higham and Tisseur's block algorithm. For `power: 1` it is
slower than the exact norm and no more accurate — the exact 1-norm of a dense
matrix is already O(n²). It earns its place when the power is greater than one,
where forming `A^k` would cost k products, and that is precisely the quantity
scaling-and-squaring needs.

It returns a **lower bound**. Exact for most matrices, rarely off by more than a
factor of two, and nothing at run time distinguishes an exact answer from an
underestimate. `tensile-diag` reports the distribution by ensemble.

`ReciprocalCondition` inherits that asymmetry: it over-estimates `1/cond`, so a
small value reliably means ill-conditioning while a large one is weaker evidence
of good conditioning. Unlike `dgecon` it needs no norm argument — the norm of
the original was captured before the factorization overwrote it, so it cannot be
given the wrong one.

---

## Workspaces and threading

A `Workspace` bundles the micro-kernel choice with the packing buffers. Both are
expensive to redo per call; the packed-B buffer alone is sized to occupy a good
fraction of L3.

```csharp
a.Multiply(b);                    // uses Workspace.Shared

using var workspace = new Workspace(multithreaded: false);
a.Multiply(b, workspace);
```

Workspace methods are internally locked, so an instance is safe to share, and
the uncontended lock is nothing against work that is at minimum O(n²). Note that
the multi-threaded GEMM already uses every core from *within* one call, so
running several concurrently usually buys nothing — give each thread its own
workspace only if you have measured that it helps.

Small problems fall back to the serial path automatically. Fork/join costs more
than it saves below roughly 4×10⁶ flops, a threshold derived from barrier cost
rather than measured.

---

## Dropping to the primitive layer

```csharp
using Tensile.Primitives;

using var dispatch = GemmDispatch.Multithreaded<Avx512Kernel16x8>();
dispatch.Multiply<Avx512Kernel16x8>(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
```

Here you choose the kernel, own the buffers, and get no argument checking. The
micro-kernel is a struct implementing `IMicroKernel` with static abstract
members, so the driver is monomorphised per kernel and every call is direct and
inlinable rather than an interface dispatch.

`Reference.Multiply` is the naive triple loop, kept as a correctness oracle.
Blocked GEMM sums the same products in a different order, so compare by residual
and never by equality.

---

## Not here yet

- **Complex**, which the transformer-winding application ultimately needs.
- **Cholesky, QR, SVD, eigenvalues.** The structure vocabulary has room for
  `SymmetricPositiveDefinite`; nothing dispatches to it yet.
- **Arithmetic for any type but `double`.** Storage is generic; operations are
  not. Adding a type is additive and breaks no signature here.
- **In-place transpose**, and a transposed GEMM. The primitive layer has no
  transpose flags at all, by design.
