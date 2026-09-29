using System.Numerics;

namespace Tensile;

/// <summary>
/// The real representation of complex matrices: an n x n complex matrix
/// X + iY becomes the 2n x 2n real matrix
///
///     [ X  -Y ]
///     [ Y   X ]
///
/// and an n x k complex panel u + iv becomes the 2n x k real stack [u; v].
///
/// The map is an algebra homomorphism -- it takes sums to sums, products to
/// products and the identity to the identity -- so it commutes with every
/// power series, and in particular exp(Embed(A)) = Embed(exp(A)). The stack
/// is compatible with it: Embed(A) * Stack(b) = Stack(A * b). Those two facts
/// are the whole mechanism by which the real <c>expm</c> and <c>expmv</c>
/// compute complex exponentials here.
///
/// It was how complex exponentials were first computed, before the library
/// had native complex products, LU and norm estimation. Now that the
/// exponentials are native it is their oracle: an independent route to the
/// same answer, built entirely on real code that is verified on its own --
/// <see cref="Expm"/> and the two <c>Expmv</c> overloads below are the
/// embedded computations, kept exactly as they shipped, for the tests, the
/// diagnostics and the benchmark that compares the two routes.
///
/// What the representation costs, for the record. The embedded matrix holds
/// four times the input's element count (twice its bytes). A product of two
/// embedded matrices does twice the flops of the complex product, because the
/// X and Y blocks are each computed twice; an embedded matrix times a stacked
/// panel does the same flops as the complex product, because the stack is not
/// redundant. The embedded 1-norm is at least the complex one and at most
/// sqrt(2) times it, since |x| + |y| &lt;= sqrt(2) * |x + iy|.
/// </summary>
internal static class ComplexEmbedding
{
    /// <summary>The 2n x 2n real matrix [[X, -Y], [Y, X]] for A = X + iY.</summary>
    /// <exception cref="ArgumentOutOfRangeException">The 2n x 2n matrix is too large to own.</exception>
    /// <exception cref="AllocationLimitException">The 2n x 2n matrix exceeds <see cref="TensileLimits.MaxElements"/>.</exception>
    public static Matrix<double> Embed(Matrix<Complex> a)
    {
        int n = a.Rows;
        var result = new Matrix<double>(2 * n, 2 * n);

        for (int j = 0; j < n; j++)
        {
            ReadOnlySpan<Complex> column = a.ReadOnlyView.Column(j);

            Span<double> left = result.Column(j);
            Span<double> right = result.Column(j + n);

            for (int i = 0; i < n; i++)
            {
                double re = column[i].Real;
                double im = column[i].Imaginary;

                left[i] = re;       // X, top-left
                left[i + n] = im;   // Y, bottom-left
                right[i] = -im;     // -Y, top-right
                right[i + n] = re;  // X, bottom-right
            }
        }

        return result;
    }

    /// <summary>The 2n x k real stack [Re B; Im B].</summary>
    /// <exception cref="ArgumentOutOfRangeException">The stack is too large to own.</exception>
    /// <exception cref="AllocationLimitException">The stack exceeds <see cref="TensileLimits.MaxElements"/>.</exception>
    public static Matrix<double> Stack(ReadOnlyMatrixView<Complex> b)
    {
        int n = b.Rows;
        var result = new Matrix<double>(2 * n, b.Columns);

        for (int j = 0; j < b.Columns; j++)
        {
            ReadOnlySpan<Complex> column = b.Column(j);
            Span<double> target = result.Column(j);

            for (int i = 0; i < n; i++)
            {
                target[i] = column[i].Real;
                target[i + n] = column[i].Imaginary;
            }
        }

        return result;
    }

    /// <summary>The inverse of <see cref="Stack"/>: a 2n x k real stack back to n x k complex.</summary>
    public static Matrix<Complex> Unstack(Matrix<double> stacked)
    {
        int n = stacked.Rows / 2;
        var result = new Matrix<Complex>(n, stacked.Columns);

        for (int j = 0; j < stacked.Columns; j++)
        {
            ReadOnlySpan<double> column = stacked.ReadOnlyView.Column(j);
            Span<Complex> target = result.Column(j);

            for (int i = 0; i < n; i++) target[i] = new Complex(column[i], column[i + n]);
        }

        return result;
    }

    /// <summary>
    /// The complex matrix a 2n x 2n real result represents, taking the
    /// orthogonal projection onto the embedded form rather than reading one
    /// block pair.
    ///
    /// A result computed in floating point is only approximately of the form
    /// [[P, -Q], [Q, P]]: rounding perturbs the two copies of P and the two
    /// copies of Q independently. Averaging the copies -- P = (E11 + E22)/2,
    /// Q = (E21 - E12)/2 -- is the orthogonal projection onto the subspace of
    /// embedded matrices, and because the exact answer lies in that subspace
    /// the projection can only move the computed result closer to it in the
    /// Frobenius norm, never further. Reading the top-left and bottom-left
    /// blocks alone would be correct too, just never better.
    /// </summary>
    public static Matrix<Complex> Project(Matrix<double> embedded)
    {
        int n = embedded.Rows / 2;
        var result = new Matrix<Complex>(n, n);

        for (int j = 0; j < n; j++)
        {
            ReadOnlySpan<double> left = embedded.ReadOnlyView.Column(j);
            ReadOnlySpan<double> right = embedded.ReadOnlyView.Column(j + n);
            Span<Complex> target = result.Column(j);

            for (int i = 0; i < n; i++)
            {
                double p = 0.5 * (left[i] + right[i + n]);
                double q = 0.5 * (left[i + n] - right[i]);
                target[i] = new Complex(p, q);
            }
        }

        return result;
    }

    /// <summary>
    /// How far a 2n x 2n real matrix is from the embedded form, as the largest
    /// disagreement between the paired blocks relative to the largest entry.
    /// Zero for an exact embedding; for a computed exponential it measures how
    /// far rounding has drifted the real algorithm off the complex structure.
    /// </summary>
    public static double StructuralDefect(Matrix<double> embedded)
    {
        int n = embedded.Rows / 2;
        double defect = 0.0;
        double scale = 0.0;

        for (int j = 0; j < n; j++)
        {
            ReadOnlySpan<double> left = embedded.ReadOnlyView.Column(j);
            ReadOnlySpan<double> right = embedded.ReadOnlyView.Column(j + n);

            for (int i = 0; i < n; i++)
            {
                defect = Math.Max(defect, Math.Abs(left[i] - right[i + n]));
                defect = Math.Max(defect, Math.Abs(left[i + n] + right[i]));

                scale = Math.Max(scale, Math.Max(Math.Abs(left[i]), Math.Abs(left[i + n])));
            }
        }

        return scale == 0.0 ? defect : defect / scale;
    }

    /// <summary>
    /// exp(A) through the embedding: the real algorithm on [[X, -Y], [Y, X]],
    /// projected back. The oracle for the native complex <c>Expm</c>; the
    /// diagnostics report the real computation, whose parameters are the ones
    /// this result used.
    /// </summary>
    public static Matrix<Complex> Expm(Matrix<Complex> a, Workspace? workspace = null, ExpmDiagnostics? diagnostics = null)
    {
        ArgumentNullException.ThrowIfNull(a);

        if (!a.IsSquare)
            throw new ArgumentException($"The matrix exponential requires a square matrix, got {a.Rows}x{a.Columns}.", nameof(a));

        if (a.Rows <= 1) return MatrixExponential.Expm(a, workspace, diagnostics);

        return Project(MatrixExponential.Expm(Embed(a), workspace, diagnostics));
    }

    /// <summary>
    /// exp(tA)B through the embedding, with the imaginary part of the trace
    /// shift removed in complex arithmetic first and restored as a rotation
    /// after -- the real algorithm's own shift can only reach the real part
    /// (finding 15). The oracle for the native complex dense <c>Expmv</c>.
    /// </summary>
    public static Matrix<Complex> Expmv(
        Matrix<Complex> a, ReadOnlyMatrixView<Complex> b, double t, ExpmvDiagnostics? diagnostics = null)
    {
        ArgumentNullException.ThrowIfNull(a);

        if (!a.IsSquare)
            throw new ArgumentException($"The exponential action requires a square matrix, got {a.Rows}x{a.Columns}.", nameof(a));

        if (b.Rows != a.Rows)
            throw new ArgumentException($"B has {b.Rows} rows, expected the operator's order {a.Rows}.", nameof(b));

        int n = a.Rows;
        if (n == 0 || b.Columns == 0) return Matrix.From(b);

        Complex trace = Complex.Zero;
        for (int i = 0; i < n; i++) trace += a[i, i];

        double omega = trace.Imaginary / n;
        var shifted = Matrix.From(a.ReadOnlyView);
        if (omega != 0.0)
            for (int i = 0; i < n; i++) shifted[i, i] += new Complex(0.0, -omega);

        var result = Unstack(MatrixExponentialAction.Expmv(Embed(shifted), Stack(b).ReadOnlyView, t, diagnostics));

        if (omega != 0.0)
        {
            Complex rotation = Complex.FromPolarCoordinates(1.0, t * omega);
            for (int j = 0; j < result.Columns; j++)
            {
                Span<Complex> column = result.Column(j);
                for (int i = 0; i < column.Length; i++) column[i] *= rotation;
            }
        }

        return result;
    }

    /// <summary>
    /// exp(tA)B for a matrix-free complex operator through the embedding, the
    /// norm bound inflated by sqrt(2) for the real representation. The oracle
    /// for the native matrix-free complex <c>Expmv</c>.
    /// </summary>
    public static Matrix<Complex> Expmv(
        ILinearOperator<Complex> op,
        ReadOnlyMatrixView<Complex> b,
        double t,
        double oneNormBound,
        ExpmvDiagnostics? diagnostics = null)
    {
        ArgumentNullException.ThrowIfNull(op);

        if (op.Order > int.MaxValue / 2)
            throw new ArgumentOutOfRangeException(nameof(op), op.Order, "The operator's real representation would not fit.");

        if (b.Rows != op.Order)
            throw new ArgumentException($"B has {b.Rows} rows, expected the operator's order {op.Order}.", nameof(b));

        if (op.Order == 0 || b.Columns == 0) return Matrix.From(b);

        return Unstack(MatrixExponentialAction.Expmv(
            new EmbeddedOperator(op), Stack(b).ReadOnlyView, t, oneNormBound * Math.Sqrt(2.0), diagnostics));
    }
}

/// <summary>
/// A complex operator presented to the real algorithms as its real
/// representation: applying this to a 2n x k stack [u; v] applies the complex
/// operator to u + iv and restacks the result. Because Embed(A) Stack(b) =
/// Stack(A b), this is exactly Embed(A) acting on the stack, and it costs one
/// application of the complex operator plus O(nk) of copying.
///
/// Holds two scratch panels, reused for as long as the column count stays the
/// same, so it is not safe to apply from two threads at once. It is created
/// per call and never escapes one, which is why that is acceptable.
/// </summary>
internal sealed class EmbeddedOperator(ILinearOperator<Complex> inner) : ILinearOperator<double>
{
    private Matrix<Complex>? _x;
    private Matrix<Complex>? _y;

    /// <inheritdoc/>
    /// <remarks>The caller has already checked that twice the inner order fits an int.</remarks>
    public int Order => 2 * inner.Order;

    /// <inheritdoc/>
    public void Apply(ReadOnlyMatrixView<double> x, MatrixView<double> y)
    {
        if (x.Rows != Order || y.Rows != Order || x.Columns != y.Columns)
        {
            throw new ArgumentException(
                $"Panels are {x.Rows}x{x.Columns} and {y.Rows}x{y.Columns}, expected {Order} rows and matching widths.",
                nameof(y));
        }

        int n = inner.Order;
        int k = x.Columns;

        if (_x is null || _x.Columns != k)
        {
            _x = new Matrix<Complex>(n, k);
            _y = new Matrix<Complex>(n, k);
        }

        for (int j = 0; j < k; j++)
        {
            ReadOnlySpan<double> from = x.Column(j);
            Span<Complex> to = _x.Column(j);
            for (int i = 0; i < n; i++) to[i] = new Complex(from[i], from[i + n]);
        }

        inner.Apply(_x.ReadOnlyView, _y!.View);

        for (int j = 0; j < k; j++)
        {
            ReadOnlySpan<Complex> from = _y.ReadOnlyView.Column(j);
            Span<double> to = y.Column(j);

            for (int i = 0; i < n; i++)
            {
                to[i] = from[i].Real;
                to[i + n] = from[i].Imaginary;
            }
        }
    }
}
