# Tensile API

Dense linear algebra for .NET. Column-major, double precision, with hand-written
micro-kernels underneath.

Everything public is in one namespace, `Tensile`, in one assembly that is
compiled with unsafe code disallowed and integer overflow checking on. The
kernels — micro-kernels, packing, blocked GEMM, LU — live in a second assembly,
`Tensile.Kernels`, whose types are all internal. There is no public pointer
anywhere, and no supported way to reach the kernels except through the types
below. What you give up is an O(1) shape check per call against work that is
at minimum O(n²); what you get is that a wrong dimension is an exception rather
than a write to someone else's memory. The reasoning is in
`docs/security-design.md`.

Zero-copy over your own storage is still available: bind a span to a shape and
the library works on it in place. See *Matrices and views*.

---

## Getting started

```csharp
using Tensile;

var a = Matrix.FromRows(new[,]
{
    { 4.0, 1.0 },
    { 1.0, 3.0 },
});

var b = Matrix.FromColumnMajor<double>(2, 1, [1.0, 2.0]);

Matrix<double> x = a.Solve(b);        // LU with partial pivoting
Matrix<double> product = a.Multiply(x);   // blocked GEMM
```

Nothing here is `IDisposable`. A `Matrix<T>` owns a pinned managed array, and a
view over it is a reference the garbage collector tracks, so storage lives
exactly as long as anything can still reach it and is reclaimed like any other
array. There is no lifetime to get wrong: no dispose-during-use, no double
free, no leak from a forgotten `using`.

---

## Matrices and views

`Matrix<T>` owns 64-byte aligned native memory in column-major order. It is
generic over `unmanaged, INumberBase<T>`, so `Matrix<float>` and
`Matrix<Half>` hold data today even though the arithmetic below is
`double`-only.

```csharp
var m = new Matrix<double>(rows: 100, columns: 40);
var z = Matrix.Zeros<double>(8, 8);
var i = Matrix.Identity<double>(8);
var f = Matrix.FromColumnMajor<double>(2, 2, [1, 2, 3, 4]);
```

A **view** is a borrowed window. `MatrixView<T>` and `ReadOnlyMatrixView<T>` are
`ref struct`s, so the compiler prevents them being stored in a field, boxed, or
captured by an async method — the usual ways a borrowed pointer outlives its
owner.

There is no way to build a view from a raw pointer. A view comes either from a
`Matrix<T>` or from **binding** a span to a `MatrixShape`, which checks that the
span is long enough:

```csharp
var shape = new MatrixShape(rows: 3, columns: 4, stride: 5);   // validates, or throws
double[] mine = new double[shape.RequiredExtent];              // (4-1)*5 + 3 = 18

MatrixView<double> view = MatrixView<double>.Bind(mine, shape);   // zero-copy over your storage
```

`MatrixShape` is the one place shape arithmetic lives, computed in `long` with
an explicit fit check — a shape whose extent would not fit an `int` cannot be
constructed. If you genuinely have a `double*`, write `new Span<double>(p, len)`
yourself: that is your `unsafe` block, correctly attributed, and it forces you to
state the length, which is exactly the fact the library needs.

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
Matrix<double> u = BuildUpperTriangular();

// Back substitution. No factorization, no branch, chosen at compile time.
Matrix<double> x = u.As<UpperTriangular>().Solve(b);
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
LuDecomposition<double> lu = a.FactorLu();

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
LuDecomposition<double> lu = a.FactorLu();     // a is not modified

Matrix<double> x  = lu.Solve(b);
Matrix<double> xa = lu.SolveAdjoint(b);         // Aᴴ x = b; Aᵀ for a real matrix

bool broken   = lu.IsSingular;                  // an exactly zero pivot
double rcond  = lu.ReciprocalCondition();
double det    = lu.Determinant();
```

`LuDecomposition<T>` is one type for every element type the library factors —
`double` and `Complex` (see [Complex matrices](#complex-matrices)). It has no
public constructor; it comes only from `FactorLu`, which exists only for element
types with arithmetic. `ReciprocalCondition` is an extension on the closed
types that have a norm estimator — `LuDecomposition<double>` and
`LuDecomposition<Complex>` — so on any other element type it would be a compile
error rather than a run-time one.

The adjoint solve replaced `SolveTransposed` when the type became generic. For a
real factorization they are the same solve; for a complex one the transpose is
rarely what is wanted and the conjugate transpose is what an adjoint operator
and the norm estimator need. (The triangular structure solves below are still
real-only and still say `SolveTransposed`.)

`FactorLu` copies, so your matrix survives, and the result keeps its own
storage alive for as long as you hold it. When the copy matters —
it is O(n²) against an O(n³) factorization, so it rarely does — factor in
place through a workspace:

```csharp
LuDecomposition<double> lu = Workspace.Shared.FactorLu(a);   // a is overwritten with the packed factors
```

The decomposition then shares `a`'s storage, which is why this takes a
`Matrix<double>` and not a view: the factors have to stay alive for as long as
the decomposition does, and a garbage-collected object can promise that where a
borrowed view cannot. Do not write to `a` while you are still using `lu`.

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

All of this works on `Matrix<Complex>` too, where ‖A‖₁ is the largest column sum
of moduli: `OneNorm`, `InfinityNorm`, `EstimateOneNorm(power)`, and
`ReciprocalCondition` on a complex factorization (the `zgecon` equivalent). The
complex estimator is the same algorithm with the adjoint Aᴴ for Aᵀ, the
direction z/|z| for the sign ±1, and no resampling of parallel sign columns —
complex signs essentially never repeat. Measured by `tensile-diag` on the complex
counterparts of the real ensembles, it is exact about as often as the real one
(27% and 41% on uniform matrices at t = 2 and 4, 60% and 75% with dominant
columns), and its worst ratio was 0.79.

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

## Matrix-free operators and norm estimation

`ILinearOperator<T>` is the extension point for anything that can be applied
but should not be formed: a matrix power, an inverse, and the operators `expm`
and `expmv` probe. It takes and returns bound views, so an implementation
receives the extents along with the data and cannot be handed a buffer shorter
than its `Order` claims. `T` is the element type — `double` or `Complex` for
anything the library computes with.

```csharp
public interface ILinearOperator<T>
{
    int Order { get; }
    void Apply(ReadOnlyMatrixView<T> x, MatrixView<T> y);            // Y := A X
}

public interface IAdjointOperator<T> : ILinearOperator<T>
{
    void ApplyAdjoint(ReadOnlyMatrixView<T> x, MatrixView<T> y);     // Y := Aᴴ X
}
```

The adjoint is a **separate capability**, not part of the base contract. A
matrix-free operator — an FEM or MTL operator assembled on the fly — can very
often apply `A` and not cheaply apply `Aᴴ`, so requiring both would force every
implementer to supply one in order that one algorithm could have it. Implement
`ILinearOperator<T>` if you only ever apply forward; `IAdjointOperator<T>` is
what `NormEstimate` asks for, because Higham and Tisseur's estimator alternates
products with `A` and its adjoint.

It is the **adjoint** — the conjugate transpose — and not the transpose,
because that is what the estimator needs. For a real operator the two are the
same, which is why this was once called `ApplyTranspose`. For a complex one the
transpose is the wrong operation, and supplying it would give an estimator that
is quietly wrong rather than one that fails.

Two implementations ship. `DenseMatrixOperator(a, power)`, an
`IAdjointOperator<double>`, applies `Aᵖ` by `p` successive panel products
without forming the power; `LuInverseOperator<T>(lu)` applies `A⁻¹` by solving,
and its adjoint by the adjoint solve, for either element type. Both are what `NormEstimate`
needs:

```csharp
double est   = NormEstimate.Of(new DenseMatrixOperator(a, power: 3)).Value;   // ≈ ‖A³‖₁
double rcond = lu.ReciprocalCondition();                                        // via LuInverseOperator<double>
```

`NormEstimate.Of` is Higham and Tisseur's block 1-norm estimator, the algorithm
behind MATLAB's `normest1`, with overloads for `IAdjointOperator<double>` and
`IAdjointOperator<Complex>`. The result is always a **lower bound**, exact on
most matrices and rarely off by more than a factor of two, and deterministic
for a given seed. It never sees the operator's entries — only the products — so
an operator that has no entries works as well as one that does.

An implementation should check that `x.Rows` and `y.Rows` equal its order and
that the two panels have the same width, and reject anything else as an
argument. The estimator always satisfies both; another caller may not.

---

## Limits

Every allocation the library makes on your behalf — storage, a result, a work
panel — goes through one path that checks the request against a process-wide
ceiling first:

```csharp
TensileLimits.MaxElements = 50_000_000;      // refuse anything over 50M elements per allocation

try
{
    var big = new Matrix<double>(10_000, 10_000);   // 100M elements
}
catch (AllocationLimitException e)
{
    Console.WriteLine($"{e.Requested} > {e.Limit}: {e.Message}");
}
```

The default is `Array.MaxLength`, the runtime's own ceiling, which is to say no
policy at all — a library should not guess your memory budget. Set it once at
startup if you are a service that would rather refuse a 16 GB request than
attempt it. `AllocationLimitException` is thrown *before* any memory is asked
for and is deliberately unrelated to `OutOfMemoryException`: one means the
library declined, the other means the runtime tried and failed, and you will
want to handle them differently.

The limit counts elements, not bytes, applies per allocation rather than in
total, and measures what you asked for — a `40×40` matrix is 1600 elements
whatever its alignment padding, though a stride larger than the row count does
count. Shape validity is checked before policy, so an impossible shape is an
`ArgumentException` under any limit.

---

## The matrix exponential

`Expm` computes exp(A) — the sum of A^k/k!, not the element-wise exponential of
the entries. The name is the one the literature uses, and is deliberately not
`Exp`, which would be indistinguishable from an element-wise map at the call
site.

```csharp
var result = a.Expm();                    // exp(A)
var result = a.Expm(workspace);           // over a workspace you own
```

The algorithm is Al-Mohy and Higham (2009) scaling-and-squaring with Padé
approximants of degree 3, 5, 7, 9 or 13, chosen by the norm of the input. It is
the 2009 algorithm and not Higham's 2005 one: the scaling parameter is picked
from estimates of ‖A^k‖^(1/k) rather than from ‖A‖, which for a nonnormal
matrix can be very much smaller. That matters because every squaring step is
another chance to amplify rounding error, so overscaling costs accuracy as well
as time — and nonnormal state matrices are exactly what this library is for.

Those ‖A^k‖^(1/k) estimates come from `NormEstimate` over a
`DenseMatrixOperator` raised to a power, so no power of A is ever formed to
measure it. This is what the operator's `power` parameter exists for.

Accuracy is **backward stable**: the computed result is the exact exponential
of A + E with ‖E‖ small relative to ‖A‖. It is not forward-accurate for every
matrix, and no algorithm for the exponential is — a matrix whose exponential is
genuinely ill-conditioned will lose digits here as it would anywhere. See Moler
and Van Loan, "Nineteen Dubious Ways to Compute the Exponential of a Matrix".

Measured against an independent Taylor oracle, the relative difference runs
from 2·10⁻¹⁶ at small norms to 4·10⁻¹³ after four squarings; `tensile-diag`
prints the table.

Cost is dominated by matrix products: three to form the powers, two more for
the degree-13 approximant, one LU factorization and solve, and one product per
squaring step — on the order of 15 to 25 products of order n.

---

## The action of the exponential

`Expmv` computes exp(tA)B without ever forming exp(tA). For a transient sweep
this is the operation you want: the cost is a few dozen applications of A to a
panel, and when B is a single vector each of those is O(n²) rather than the
O(n³) of a matrix product.

```csharp
var y = a.Expmv(b.ReadOnlyView);              // exp(A) B
var y = a.Expmv(b.ReadOnlyView, t: 2.5);      // exp(2.5 A) B, t may be negative or zero
```

Al-Mohy and Higham (2011). The truncation degree m and the scaling s are chosen
together to minimise m·s — the number of applications — subject to a backward
error bound, and the inner loop stops early once the remaining terms cannot
change the result at working precision, so m is a cap rather than a count.

### How much it saves, and when it stops saving

Counted rather than timed, so these numbers are machine independent;
`tensile-diag` prints the table. Ratio is expm flops over expmv flops, with B
a single column unless stated:

| n | ‖A‖₁ | n₀ | applications | flop ratio vs `Expm` |
| --- | --- | --- | --- | --- |
| 64 | 1 | 1 | 12 | 34× |
| 256 | 1 | 1 | 10 | 163× |
| 1000 | 10 | 1 | 26 | 283× |
| 1000 | 100 | 1 | 30 | 378× |
| 256 | 10 | 8 | 17 | 13.9× |
| 256 | 10 | 64 | 17 | **1.8×** |

**The advantage is roughly n/n₀, so it is large exactly when B is narrow and
it disappears when B is wide.** At n=256 it falls from 163× at one column to
1.8× at sixty-four. If you need exp(A) applied to many columns, form the
exponential once with `Expm` and multiply — that is what the last row is
telling you.

Two caveats on reading the ratio. It is **flops, not time**: `Expm` spends its
flops in GEMM, which runs near the machine's peak, while `Expmv` spends them in
unpacked panel products, which are memory-bound and run at a fraction of it, so
the wall-clock advantage is smaller. And no benchmark has been taken — that
needs the verification machine.

### Matrix-free operators

The overload that matters for a real FEM or MTL model takes an operator rather
than a matrix, and uses **only** `Apply` — no transpose, no entries, no trace:

```csharp
var y = MatrixExponentialAction.Expmv(op, b.ReadOnlyView, t: 1.0, oneNormBound: bound);
```

This is why `ILinearOperator<T>` does not require an adjoint. The price is the
parameter choice: without products by the adjoint the ‖A^p‖^(1/p) quantities
cannot be estimated, so the scaling falls back to the bound you supply. Since
‖A^p‖^(1/p) ≤ ‖A‖, that is always safe — it can only pick a larger s than
necessary, never a smaller one — but for a strongly nonnormal operator it can
be a lot more work than the dense path would do. Supply the tightest bound you
have.

---

## Complex matrices

### Products

`Matrix<Complex>` (`System.Numerics.Complex`) multiplies with the same call
shapes as the real type. Complex scalars have no compile-time constant form, so
the scaled accumulation is a separate overload rather than optional arguments:

```csharp
var c = a.Multiply(b);                                   // A B
a.MultiplyInto(b, c.View);                               // c := A B
a.MultiplyInto(b, c.View, alpha, beta);                  // c := beta c + alpha A B
workspace.Multiply(a.ReadOnlyView, b.ReadOnlyView, c.View, alpha, beta);
```

These are native, not embedded: the **4M method**, four real products on the
workspace's own kernel and dispatch — Re(AB) = ArBr − AiBi, Im(AB) = ArBi + AiBr.
That is exactly the flop count of a conventional complex product, and it runs
on the real GEMM that is at parity with BLIS. The destination is written only
after all four products succeed, so a failure leaves it untouched.

It is 4M and not the cheaper 3M (three products) because 3M's error in the
imaginary part is relative to the whole magnitude of the operands, so a small
imaginary part is swamped. Measured on operands whose imaginary parts are a
factor ρ smaller than their real parts:

| ρ | 4M, relative error of Im(AB) | 3M, relative error of Im(AB) |
| --- | --- | --- |
| 10³ | 6e-14 | 9e-11 |
| 10⁶ | 5e-14 | 1e-7 |
| 10⁹ | 3e-14 | 1e-4 |
| 10¹² | 9e-14 | 0.19 |

This matters for line models specifically: in a product such as
ZY = (R + jωL)(G + jωC), the loss terms ω(LG + RC) land in the small imaginary
part, next to a dominant −ω²LC. 3M would lose the damping first.

The cost of 4M is splitting: real and imaginary copies of both operands and of
the product. The workspace keeps those buffers between products, so a run of
products — the squarings of an exponential, the trailing updates of a
factorization — allocates them once. Retention is capped at 2²¹ elements
(16 MiB, a 512×512 complex product in full) because the buffers grow with the
problem and `Workspace.Shared` lives as long as the process; a larger product
gets buffers of its own for that call, and a workspace that has run complex
products may hold up to that much more memory than one that has not. Disposing
a workspace releases them.

With the buffers retained, 4M measured 4.4× one real product at n=128 and 4.0×
at n=512 on a development container, against an ideal 4.0, and 1.8–2.0× faster
than the embedded route. Those are noisy figures from a short job; the
verification-machine measurement is outstanding. `ComplexGemmBenchmarks` is the
instrument.

### LU and solves

Complex matrices factor and solve with the same calls as real ones:

```csharp
LuDecomposition<Complex> lu = z.FactorLu();   // z is not modified
Matrix<Complex> x = lu.Solve(b);              // Z x = b
Matrix<Complex> y = lu.SolveAdjoint(b);       // Zᴴ y = b, the conjugate transpose
Complex det = lu.Determinant();

Matrix<Complex> x2 = z.Solve(b);              // factor and discard
```

Partial pivoting picks the entry with the largest |Re| + |Im|, not the largest
modulus. That is LAPACK's `izamax` rule, so a pivot sequence can be compared
with one from `zgetrf`; it is within √2 of the modulus, so it carries the same
stability, and it needs no square root.

The algorithm is the real one, blocked the same way, written once over the
element type. Its trailing updates — nearly all of the work — are 4M products on
the workspace's real GEMM. The panels and triangular solves are safe managed
code, slower per flop than the real kernel's but O(n²·nb) of the total.

`lu.ReciprocalCondition()` works on a complex factorization as it does on a
real one (see [Norms and conditioning](#norms-and-conditioning)). What it does
not have yet is a **measured block size**: the default is the real
factorization's, which was measured for real; whether complex wants a different
one is open.

### Exponentials

`Expm` and `Expmv` accept `Matrix<Complex>` with the same call shape as the
real ones:

```csharp
var e = z.Expm();                          // exp(Z)
var y = z.Expmv(b.ReadOnlyView, t: 0.5);   // exp(0.5 Z) B, t real
```

In the standard multiconductor-line formulation this is the chain-parameter
matrix Φ(ℓ) = exp(Mℓ), with M = [[0, −Z(ω)], [−Y(ω), 0]] complex at each
frequency, so a frequency sweep is one of these per frequency.

**How they are computed.** Natively: the same algorithms as the real
overloads — Al-Mohy and Higham's scaling and squaring for `Expm`, their
truncated Taylor action for `Expmv` — written once over the element type and
run in complex arithmetic. Products are 4M complex products on the workspace's
GEMM, the Padé denominator is factored by the complex LU, and the
‖Aᵏ‖^(1/k) estimates come from the complex norm estimator. For `double` the
same code performs exactly the operations the real-only implementation did.

They were first computed through the real representation of the complex
matrix — X + iY as the 2n×2n real [[X, −Y], [Y, X]], which the exponential
commutes with — and that route is kept, internally, as the oracle the native
one is tested against. Against it, the native route does half the flops for
`Expm`, needs half the memory, and chooses its scaling from the complex norms
rather than from real ones up to √2 larger; on random complex matrices
`tensile-diag` shows it never taking more squarings, and sometimes one fewer.
On a development container (noisy; within-run ratios) native `Expm` measured
1.26× faster than the embedded route at n=64 and 1.49× at n=256, and native
`Expmv` level at n=64 and 1.47× faster at n=256. Not yet measured on the
verification machine; `ComplexExponentialBenchmarks` is the instrument.

`Expmv` removes the whole trace shift μ = trace(A)/n, real and imaginary parts
together, and restores it one factor exp(tμ/s) per scaling step. The imaginary
part is the one that matters for an operator carrying a large jωI term; the
embedded route could only remove it by special handling, and without that it
cost up to 24× the work in measurement. (The MTL chain matrix has zero trace,
so it is unaffected either way.)

A matrix-free complex operator — an `ILinearOperator<Complex>` — has its own
overload:

```csharp
var y = MatrixExponentialAction.Expmv(op, b.ReadOnlyView, t: 1.0, oneNormBound: bound);
```

As with the real matrix-free overload, the scaling comes from the bound you
supply, an upper bound on the largest column sum of moduli. There is no trace
to shift by either, so an operator carrying a large jωI term keeps it.
If you know that shift, remove it yourself: exp(tA)B = e^(iωt)·exp(t(A − iωI))B,
exactly, and apply the operator without it.

---

## Not here yet

- **A public complex dense operator.** `EstimateOneNorm` on a complex matrix
  uses an internal one; `DenseMatrixOperator` is real-only.
- **Cholesky, QR, SVD, eigenvalues.** The structure vocabulary has room for
  `SymmetricPositiveDefinite`; nothing dispatches to it yet.
- **Arithmetic for any type but `double` and `Complex`.** Storage is generic;
  operations exist for those two. LU is written once over the element type, so
  a third type needs its element kernels, not a new factorization.
- **In-place transpose**, and a transposed GEMM. The primitive layer has no
  transpose flags at all, by design.
