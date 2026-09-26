using System.Numerics;

namespace Tensile;

/// <summary>
/// The arithmetic an algorithm needs from its element type, as static members
/// so that an algorithm written once over <c>T</c> binds to the right
/// implementation at compile time.
///
/// This is how the library keeps "one algorithm, every numeric type" without
/// paying for it. An algorithm takes a type parameter
/// <c>TKernels : struct, IElementKernels&lt;T&gt;</c> alongside <c>T</c> and
/// calls <c>TKernels.Multiply</c>; the JIT compiles a separate copy of the
/// algorithm for each struct type argument, so every call is direct and
/// inlinable, with no virtual dispatch and no per-element cost. It is the same
/// device the library already uses twice: <c>IMicroKernel</c> selects the
/// GEMM micro-kernel this way, and <see cref="IMatrixStructure"/> selects the
/// solve.
///
/// The members split into two kinds, and the split is the reason this is an
/// interface rather than a reliance on <see cref="INumberBase{TSelf}"/>. Some
/// operate on elements (<see cref="Multiply"/>, below). Others produce the real
/// quantities every algorithm here is steered by -- norms, magnitudes -- and
/// those are <see cref="double"/> whatever <c>T</c> is. Generic math cannot
/// express that second kind: for <see cref="Complex"/>,
/// <c>INumberBase&lt;Complex&gt;.Abs</c> returns a <see cref="Complex"/>, not
/// a magnitude, so an algorithm written against <c>INumberBase</c> alone has
/// no way to compare a norm with a threshold.
///
/// Members arrive as the algorithms that need them do, each with an
/// implementation for every element type at once, so the interface is shaped
/// by more than one case rather than guessed from one. LU and the norm
/// estimator's products join it with complex LU and complex <c>normest1</c>.
/// Note that pivoting will bring its own measure: LAPACK's <c>izamax</c>
/// pivots on |Re| + |Im|, not on <see cref="Magnitude"/>, and matching it is
/// what keeps pivot sequences comparable with <c>zgetrf</c>.
///
/// Internal: the public surface stays concrete, with overloads only for the
/// element types that implement this, so an element type with no arithmetic
/// fails to compile rather than at run time.
/// </summary>
/// <typeparam name="T">The element type.</typeparam>
internal interface IElementKernels<T> where T : unmanaged, INumberBase<T>
{
    /// <summary>
    /// C := beta*C + alpha*A*B. Shapes are validated, and a zero beta
    /// overwrites rather than scales, so a C full of NaN still yields a finite
    /// result -- the contract of <see cref="Workspace.Multiply(ReadOnlyMatrixView{double}, ReadOnlyMatrixView{double}, MatrixView{double}, double, double)"/>
    /// for every element type.
    /// </summary>
    /// <exception cref="ArgumentException">The shapes are not conformable.</exception>
    static abstract void Multiply(
        Workspace workspace, ReadOnlyMatrixView<T> a, ReadOnlyMatrixView<T> b, MatrixView<T> c, T alpha, T beta);

    /// <summary>|x|, the modulus, as a real number.</summary>
    static abstract double Magnitude(T value);

    /// <summary>||A||_1, the largest column sum of magnitudes. Exact, O(m*n).</summary>
    static abstract double OneNorm(ReadOnlyMatrixView<T> a);

    /// <summary>||A||_inf, the largest row sum of magnitudes. Exact, O(m*n).</summary>
    static abstract double InfinityNorm(ReadOnlyMatrixView<T> a);
}

/// <summary>
/// <see cref="IElementKernels{T}"/> for <see cref="double"/>: each member is
/// the existing real path, unchanged. Nothing about the double arithmetic moves
/// when an algorithm goes through this rather than calling it directly.
/// </summary>
internal readonly struct DoubleKernels : IElementKernels<double>
{
    /// <inheritdoc/>
    public static void Multiply(
        Workspace workspace,
        ReadOnlyMatrixView<double> a,
        ReadOnlyMatrixView<double> b,
        MatrixView<double> c,
        double alpha,
        double beta) =>
        workspace.Multiply(a, b, c, alpha, beta);

    /// <inheritdoc/>
    public static double Magnitude(double value) => Math.Abs(value);

    /// <inheritdoc/>
    public static double OneNorm(ReadOnlyMatrixView<double> a) => a.OneNorm();

    /// <inheritdoc/>
    public static double InfinityNorm(ReadOnlyMatrixView<double> a) => a.InfinityNorm();
}
