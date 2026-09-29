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
/// by more than one case rather than guessed from one. LU brought
/// <see cref="PivotMagnitude"/> and <see cref="Conjugate"/>; the norm
/// estimator brought <see cref="Sign"/> and <see cref="SignsAreDiscrete"/>;
/// the exponentials brought <see cref="Scale"/>, <see cref="Exp"/>, and the
/// two members that are not arithmetic at all, <see cref="PowerOperator"/> and
/// <see cref="FactorLuInPlace"/> -- the element type's own operator and
/// factorization, which an algorithm written over <c>T</c> has no other way to
/// reach without testing what <c>T</c> is.
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

    /// <summary>
    /// The measure partial pivoting maximises. For a real number it is |x|.
    /// For a complex one it is |Re| + |Im|, not the modulus: that is what
    /// LAPACK's <c>izamax</c> uses (its <c>cabs1</c>), and matching it keeps a
    /// pivot sequence comparable with <c>zgetrf</c>'s. It is within a factor
    /// of sqrt(2) of the modulus, so the growth bound of partial pivoting
    /// survives, and it needs no square root.
    /// </summary>
    static abstract double PivotMagnitude(T value);

    /// <summary>The complex conjugate; the identity for a real type. What turns a transpose into an adjoint.</summary>
    static abstract T Conjugate(T value);

    /// <summary>
    /// The direction of <paramref name="value"/>, of unit modulus, with the
    /// direction of zero taken as 1: +/-1 for a real number, z/|z| for a
    /// complex one. The norm estimator's sign matrix is made of these.
    /// </summary>
    static abstract T Sign(T value);

    /// <summary>
    /// Whether <see cref="Sign"/> takes values in a finite set (+/-1), so that
    /// two sign vectors can be exactly parallel and the norm estimator's
    /// resampling of repeated columns means something. True for a real type.
    /// </summary>
    static abstract bool SignsAreDiscrete { get; }

    /// <summary>
    /// <paramref name="value"/> times a real <paramref name="factor"/>. For a
    /// complex value, each part is scaled on its own: a complex product with
    /// (factor + 0i) would compute the same thing with extra roundings that
    /// happen to be exact, and a NaN from 0 * infinity in the one case where
    /// they are not.
    /// </summary>
    static abstract T Scale(T value, double factor);

    /// <summary>The scalar exponential.</summary>
    static abstract T Exp(T value);

    /// <summary>
    /// A dense matrix raised to a power, as an operator with an adjoint: what
    /// the norm estimator is pointed at to reach ||A^p||_1 without forming A^p.
    /// </summary>
    static abstract IAdjointOperator<T> PowerOperator(Matrix<T> a, int power);

    /// <summary>
    /// Factor <paramref name="a"/> in place on <paramref name="workspace"/>,
    /// through the element type's own LU: the kernel factorization for a real
    /// type, the generic blocked one for complex.
    /// </summary>
    static abstract LuDecomposition<T> FactorLuInPlace(Workspace workspace, Matrix<T> a);

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
    public static double PivotMagnitude(double value) => Math.Abs(value);

    /// <inheritdoc/>
    public static double Conjugate(double value) => value;

    /// <inheritdoc/>
    /// <remarks>Anything not at least zero, NaN included, is taken as negative -- the rule the real estimator always had.</remarks>
    public static double Sign(double value) => value >= 0.0 ? 1.0 : -1.0;

    /// <inheritdoc/>
    public static bool SignsAreDiscrete => true;

    /// <inheritdoc/>
    public static double Scale(double value, double factor) => value * factor;

    /// <inheritdoc/>
    public static double Exp(double value) => Math.Exp(value);

    /// <inheritdoc/>
    public static IAdjointOperator<double> PowerOperator(Matrix<double> a, int power) => new DenseMatrixOperator(a, power);

    /// <inheritdoc/>
    public static LuDecomposition<double> FactorLuInPlace(Workspace workspace, Matrix<double> a) => workspace.FactorLu(a);

    /// <inheritdoc/>
    public static double OneNorm(ReadOnlyMatrixView<double> a) => a.OneNorm();

    /// <inheritdoc/>
    public static double InfinityNorm(ReadOnlyMatrixView<double> a) => a.InfinityNorm();
}
