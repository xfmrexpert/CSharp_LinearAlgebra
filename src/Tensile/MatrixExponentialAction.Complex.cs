using System.Numerics;

namespace Tensile;

// The complex exponential action, through the real embedding for now. As with
// the complex expm, this is a partial so that a native implementation can
// replace the body without moving the public signature.
public static partial class MatrixExponentialAction
{
    /// <summary>
    /// exp(tA)B for a complex A and B, without forming the exponential.
    ///
    /// Computed today by running the real algorithm on the 2n x 2n real
    /// representation [[X, -Y], [Y, X]] of A = X + iY, applied to the stacked
    /// panel [Re B; Im B]. Unlike the complex <c>Expm</c>, this costs no extra
    /// flops: an embedded matrix times a stacked panel does exactly the work
    /// of the complex product. It does move twice the bytes of the matrix per
    /// application, and the panel path is memory-bound, so expect the time to
    /// sit nearer twice a native implementation's than level with it
    /// (unmeasured).
    ///
    /// The trace shift is split in two. The real representation's trace is
    /// 2*Re(trace A), so the real algorithm's own shift can only ever remove
    /// Re(mu), mu = trace(A)/n. The imaginary part is therefore removed here,
    /// in complex arithmetic, before embedding, and restored afterwards as a
    /// single factor e^(i t Im mu). That factor has unit modulus, so unlike
    /// the real part it cannot overflow or underflow however large t Im mu is.
    /// Without this, an operator carrying a large imaginary diagonal -- a
    /// j*omega*I term -- keeps it in the matrix: measured, adding 100i*I to a
    /// skew-Hermitian operator of norm 5 took the work from 23 applications
    /// to 561.
    ///
    /// One cost remains, in the parameter choice and not in the answer: the
    /// embedded norms can exceed the complex ones by up to sqrt(2), which can
    /// raise the scaling s. Random complex matrices sit near that bound, at
    /// 1.28 to 1.39 in measurement; real ones sit at exactly 1.
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
    /// <exception cref="ArgumentOutOfRangeException">
    /// The real representation is too large to own. The 2n x 2n matrix has
    /// four times the input's element count, so it can exceed a size the input
    /// did not; so can the 2n-row stacked panel.
    /// </exception>
    /// <exception cref="AllocationLimitException">
    /// The real representation exceeds <see cref="TensileLimits.MaxElements"/>,
    /// which for the same reason can happen when the input itself fits.
    /// </exception>
    public static Matrix<Complex> Expmv(
        this Matrix<Complex> a, ReadOnlyMatrixView<Complex> b, double t = 1.0) =>
        Expmv(a, b, t, diagnostics: null);

    /// <summary>
    /// The complex implementation. The diagnostics report the real
    /// computation on the embedding, whose parameters are the ones the complex
    /// result actually used.
    /// </summary>
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

        // exp(tA) = e^(i t omega) exp(t (A - i omega I)), omega = Im(trace A)/n,
        // exactly, since a multiple of the identity commutes with A. The real
        // part of the shift is left in: the real algorithm removes it itself
        // and folds it back one factor per scaling step, which is what keeps
        // that factor and the shifted result from overflowing.
        double omega = Trace(a).Imaginary / n;
        var shifted = omega == 0.0 ? a : ShiftDiagonal(a, new Complex(0.0, -omega));

        var embedded = ComplexEmbedding.Embed(shifted);
        var stacked = ComplexEmbedding.Stack(b);

        var result = ComplexEmbedding.Unstack(Expmv(embedded, stacked.ReadOnlyView, t, diagnostics));

        if (omega != 0.0) ScaleInPlace(result, Complex.FromPolarCoordinates(1.0, t * omega));

        return result;
    }

    /// <summary>
    /// exp(tA)B for a complex operator that is applied rather than stored --
    /// the form a frequency-domain FEM or MTL operator plugs into.
    ///
    /// Only <see cref="ILinearOperator{T}.Apply"/> is used: no adjoint, no
    /// entries, no trace. The operator is presented to the real algorithm as
    /// its 2n x 2n real representation, applied through the complex operator
    /// itself, so each application costs one complex application plus O(nk)
    /// of copying.
    ///
    /// Two consequences of having no entries. The scaling falls back to
    /// <paramref name="oneNormBound"/>, as the real matrix-free overload's
    /// does; it is multiplied by sqrt(2) internally, because that is the most
    /// the real representation's norm can exceed the complex one by. And
    /// there is no trace, so no shift of either kind: an operator carrying a
    /// large j*omega*I term keeps it, and it can cost an order of magnitude in
    /// applications. If you know the shift, remove it yourself --
    /// exp(tA)B = e^(i omega t) exp(t(A - i omega I))B exactly -- and apply
    /// the operator without it.
    /// </summary>
    /// <param name="op">The complex operator. Applied, never inspected.</param>
    /// <param name="b">The panel to apply exp(tA) to, with as many rows as the operator's order.</param>
    /// <param name="t">The time, or any real multiplier on A. May be negative or zero.</param>
    /// <param name="oneNormBound">An upper bound on the complex ||A||_1. Must be finite and not negative.</param>
    /// <returns>A new matrix holding exp(tA)B, the same shape as B.</returns>
    /// <exception cref="ArgumentNullException">The operator is null.</exception>
    /// <exception cref="ArgumentException">B has the wrong number of rows.</exception>
    /// <exception cref="ArgumentOutOfRangeException">
    /// The norm bound is negative or not finite, or the operator's order is so
    /// large that its real representation's order, twice it, would not fit an
    /// int. An operator is free to report any order, so that is checked before
    /// anything is sized from it.
    /// </exception>
    /// <exception cref="AllocationLimitException">A panel of the real representation exceeds <see cref="TensileLimits.MaxElements"/>.</exception>
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

        // Before anything is sized from the order: an operator reports what it
        // likes, and 2 * Order would trap in this checked assembly rather than
        // be reported as the argument error it is.
        if (op.Order > int.MaxValue / 2)
        {
            throw new ArgumentOutOfRangeException(
                nameof(op), op.Order, "The operator's real representation would have an order that does not fit an int.");
        }

        if (b.Rows != op.Order)
            throw new ArgumentException($"B has {b.Rows} rows, expected the operator's order {op.Order}.", nameof(b));

        if (!double.IsFinite(oneNormBound) || oneNormBound < 0.0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(oneNormBound), oneNormBound, "The norm bound must be finite and not negative.");
        }

        if (op.Order == 0 || b.Columns == 0) return Matrix.From(b);

        var stacked = ComplexEmbedding.Stack(b);
        var result = Expmv(
            new EmbeddedOperator(op), stacked.ReadOnlyView, t, oneNormBound * Math.Sqrt(2.0), diagnostics);

        return ComplexEmbedding.Unstack(result);
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

    private static void ScaleInPlace(Matrix<Complex> a, Complex scale)
    {
        for (int j = 0; j < a.Columns; j++)
        {
            Span<Complex> column = a.Column(j);
            for (int i = 0; i < column.Length; i++) column[i] *= scale;
        }
    }
}
