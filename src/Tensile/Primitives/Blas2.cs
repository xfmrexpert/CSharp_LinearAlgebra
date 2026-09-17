namespace Tensile.Primitives;

/// <summary>
/// Products of a square matrix with a narrow panel of columns.
///
/// These exist rather than routing through <see cref="Gemm"/> because the
/// 1-norm estimator probes with t columns, where t is 2 or 4. Packing an
/// n x n operand into micro-panels to multiply it by a 2-column matrix costs
/// more in packing than the multiply itself saves, and the estimator's whole
/// value is that each product is O(n^2 t) rather than O(n^3).
///
/// Both directions are written so the contiguous dimension of column-major
/// storage is the one being streamed: A*X accumulates axpys over columns of A,
/// A^T*X takes inner products down columns of A. Neither needs A transposed in
/// memory, which is why the "no transposes" rule in the GEMM path costs nothing
/// here.
/// </summary>
internal static unsafe class Blas2
{
    /// <summary>Y := A * X, A being n x n column-major, X and Y being n x t.</summary>
    public static void Multiply(
        int n, int t, double* a, int lda, double* x, int ldx, double* y, int ldy)
    {
        for (int col = 0; col < t; col++)
        {
            double* source = x + (nint)col * ldx;
            double* target = y + (nint)col * ldy;

            new Span<double>(target, n).Clear();

            for (int j = 0; j < n; j++)
            {
                double scale = source[j];
                if (scale != 0.0) Blas1.Axpy(n, scale, a + (nint)j * lda, target);
            }
        }
    }

    /// <summary>Y := A^T * X, A being n x n column-major, X and Y being n x t.</summary>
    public static void MultiplyTransposed(
        int n, int t, double* a, int lda, double* x, int ldx, double* y, int ldy)
    {
        for (int col = 0; col < t; col++)
        {
            double* source = x + (nint)col * ldx;
            double* target = y + (nint)col * ldy;

            for (int j = 0; j < n; j++)
                target[j] = Blas1.Dot(n, a + (nint)j * lda, source);
        }
    }
}
