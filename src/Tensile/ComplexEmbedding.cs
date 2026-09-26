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
/// This exists for two reasons, one temporary and one permanent. Until the
/// library has native complex primitives (complex GEMM, complex LU, a complex
/// norm estimator) it is how complex exponentials are computed at all. After
/// that it remains the oracle for the native path: an independent route to the
/// same answer, built entirely on real code that is already verified.
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
}
