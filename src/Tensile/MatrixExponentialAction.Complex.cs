using System.Numerics;

namespace Tensile;

// The complex exponential action. As with the complex expm, a partial, so the
// implementation could move from the real embedding to native complex
// arithmetic without the public signatures moving. The embedded route is kept
// as the oracle, in ComplexEmbedding.
public static partial class MatrixExponentialAction
{
    /// <summary>
    /// exp(tA)B for a complex A and B, without forming the exponential.
    ///
    /// The same algorithm as the real overload, in complex arithmetic. The
    /// trace shift is complex too: mu = trace(A)/n is removed whole, real and
    /// imaginary parts together, and restored one factor exp(t*mu/s) per
    /// scaling step. The imaginary part is the one that matters for an
    /// oscillatory operator -- a j*omega*I term -- and removing it is what the
    /// first, embedded version of this method could not do by itself (see
    /// CLAUDE.md finding 15). The ||A^p||^(1/p) estimates come from the
    /// complex norm estimator on the complex matrix, so they are not inflated
    /// by the up-to-sqrt(2) that the real representation's norms carried.
    ///
    /// For an operator that is applied rather than stored, see the
    /// <see cref="Expmv(ILinearOperator{Complex}, ReadOnlyMatrixView{Complex}, double, double)"/>
    /// overload.
    /// </summary>
    /// <param name="a">The operator. Must be square. Not modified.</param>
    /// <param name="b">The panel to apply exp(tA) to, with as many rows as A. Not modified.</param>
    /// <param name="t">The time, or any real multiplier on A. May be negative or zero.</param>
    /// <returns>A new matrix holding exp(tA)B, the same shape as B.</returns>
    /// <exception cref="ArgumentNullException">The matrix is null.</exception>
    /// <exception cref="ArgumentException">A is not square, or B has the wrong number of rows.</exception>
    public static Matrix<Complex> Expmv(
        this Matrix<Complex> a, ReadOnlyMatrixView<Complex> b, double t = 1.0) =>
        Expmv(a, b, t, diagnostics: null);

    /// <summary>The complex dense implementation, with an optional report of the parameters chosen.</summary>
    internal static Matrix<Complex> Expmv(
        Matrix<Complex> a,
        ReadOnlyMatrixView<Complex> b,
        double t,
        ExpmvDiagnostics? diagnostics)
    {
        ArgumentNullException.ThrowIfNull(a);

        if (!a.IsSquare)
            throw new ArgumentException($"The exponential action requires a square matrix, got {a.Rows}x{a.Columns}.", nameof(a));

        if (b.Rows != a.Rows)
            throw new ArgumentException($"B has {b.Rows} rows, expected the operator's order {a.Rows}.", nameof(b));

        int n = a.Rows;
        if (n == 0 || b.Columns == 0) return Matrix.From(b);

        // Each part divided by n on its own: a complex division by (n + 0i)
        // would reach the same values by a longer route.
        Complex trace = Trace(a);
        var mu = new Complex(trace.Real / n, trace.Imaginary / n);
        var shifted = ShiftDiagonal(a, -mu);

        return Dense<Complex, ComplexKernels>(shifted, b, t, mu, diagnostics);
    }

    /// <summary>
    /// exp(tA)B for a complex operator that is applied rather than stored --
    /// the form a frequency-domain FEM or MTL operator plugs into.
    ///
    /// Only <see cref="ILinearOperator{T}.Apply"/> is used: no adjoint, no
    /// entries, no trace. So, as for the real matrix-free overload, the
    /// scaling falls back to <paramref name="oneNormBound"/> and there is no
    /// shift. An operator carrying a large j*omega*I term keeps it, and it
    /// can cost an order of magnitude in applications; if you know the shift,
    /// remove it yourself -- exp(tA)B = e^(i omega t) exp(t(A - i omega I))B
    /// exactly -- and apply the operator without it.
    /// </summary>
    /// <param name="op">The complex operator. Applied, never inspected.</param>
    /// <param name="b">The panel to apply exp(tA) to, with as many rows as the operator's order.</param>
    /// <param name="t">The time, or any real multiplier on A. May be negative or zero.</param>
    /// <param name="oneNormBound">An upper bound on ||A||_1, the largest column sum of moduli. Must be finite and not negative.</param>
    /// <returns>A new matrix holding exp(tA)B, the same shape as B.</returns>
    /// <exception cref="ArgumentNullException">The operator is null.</exception>
    /// <exception cref="ArgumentException">B has the wrong number of rows.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The norm bound is negative or not finite.</exception>
    public static Matrix<Complex> Expmv(
        ILinearOperator<Complex> op, ReadOnlyMatrixView<Complex> b, double t, double oneNormBound) =>
        Expmv(op, b, t, oneNormBound, diagnostics: null);

    /// <summary>The matrix-free complex implementation, with an optional report of the parameters chosen.</summary>
    internal static Matrix<Complex> Expmv(
        ILinearOperator<Complex> op,
        ReadOnlyMatrixView<Complex> b,
        double t,
        double oneNormBound,
        ExpmvDiagnostics? diagnostics)
    {
        ArgumentNullException.ThrowIfNull(op);

        if (b.Rows != op.Order)
            throw new ArgumentException($"B has {b.Rows} rows, expected the operator's order {op.Order}.", nameof(b));

        if (!double.IsFinite(oneNormBound) || oneNormBound < 0.0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(oneNormBound), oneNormBound, "The norm bound must be finite and not negative.");
        }

        if (op.Order == 0 || b.Columns == 0) return Matrix.From(b);

        return Run<Complex, ComplexKernels>(op, b, t, Complex.Zero, oneNormBound, alpha: null, diagnostics);
    }

    private static Complex Trace(Matrix<Complex> a)
    {
        Complex total = Complex.Zero;
        for (int i = 0; i < a.Rows; i++) total += a[i, i];
        return total;
    }

    /// <summary>A copy of <paramref name="a"/> with <paramref name="delta"/> added to its diagonal.</summary>
    private static Matrix<Complex> ShiftDiagonal(Matrix<Complex> a, Complex delta)
    {
        var result = Matrix.From(a.ReadOnlyView);
        for (int i = 0; i < a.Rows; i++) result[i, i] += delta;
        return result;
    }
}
